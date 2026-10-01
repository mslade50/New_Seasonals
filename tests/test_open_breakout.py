"""Offline lifecycle and boundary tests. No IBKR connection or paid data required."""
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import replace
import pandas as pd
import pytest
from open_breakout.config import Config
from open_breakout.inputs import calendar_dates, legacy_score, session_range, build_manifest, load_manifest, digest
from open_breakout.strategy import State, on_price, size_order, NY
from open_breakout.store import Store
from open_breakout.service import Service
from open_breakout.replay import SimBroker

ROOT=Path(__file__).resolve().parents[1]
DAY='2026-09-24'

def at(text):return datetime.fromisoformat(DAY+'T'+text).replace(tzinfo=NY)

@pytest.fixture
def config(tmp_path):
    raw=json.loads((ROOT/'config/open_breakout.example.json').read_text(encoding='utf-8-sig'))
    raw['account']='DU_TEST'
    raw['entry_order_type']='ioc' # Retain the historical IOC regression suite.
    for i,m in enumerate(raw['markets']):
        for j,key in enumerate(['signal','execution']):
            m[key].update(con_id=100+i*2+j,expiry='20261218')
    p=tmp_path/'config.json';p.write_text(json.dumps(raw))
    return Config.load(p)

def manifest(config):
    b=dict(day=DAY,prepared_at=at('09:00:00').isoformat(),previous_session='2026-09-23',
           config_hash=config.fingerprint,risk_series='legacy_63d_sma10',score=25,roll_verified=True,
           markets={m.name:dict(prior_tr=40.,signal_con_id=m.signal.con_id,execution_con_id=m.execution.con_id) for m in config.markets})
    return {**b,'hash':digest(b)}

def setup_service(config,tmp_path,broker_class=SimBroker):
    now=[at('09:30:00')]
    broker=broker_class(config,lambda:now[0])
    store=Store(tmp_path/'session.sqlite',config.fingerprint,DAY)
    service=Service(config,manifest(config),store,broker,lambda:now[0])
    return service,broker,store,now


def test_operator_control_preserves_exits_and_disarms_only_selected_market(config,tmp_path):
    from types import SimpleNamespace as NS
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            await tick(service,b,now,20010,'09:30:01')
            s=service.states['NQ']
            original=len(store.orders())
            selected=NS(contract=NS(conId=config.markets[0].execution.con_id),
                        order=NS(orderId=s.stop_order,permId=101))
            service.operator_order(selected)
            assert service.manual_markets=={'NQ'} and not service.halted
            assert len(store.orders())==original and service.states['ES'].phase=='FLAT'
            assert config.markets[0].execution.con_id not in service._expected()
            service._stop_dead('NQ',s.stop_order,'manual edit',True)
            await service._emergency_flatten('NQ','already queued before edit')
            await tick(service,b,now,19900,'09:31:00')
            service.fill(s.stop_order,'manual-fill',1,19990)
            assert len(store.orders())==original and not service.flattened
            assert store.day_r(['NQ'])['NQ']['r'] is None
            assert 'manual control' in store.day_r(['NQ'])['NQ']['note']
            assert store.get('manual_markets')==['NQ']
        finally:store.close()
    asyncio.run(run())

async def tick(service,broker,now,price,stamp,market='NQ'):
    now[0]=at(stamp)
    broker.update_quote(market,price-.25,price,now[0])
    service.tick(market,now[0],price)
    await service.drain()


def test_resting_entries_are_held_before_crossing(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            entries=[o for o in store.orders().values() if o['role']=='ENTRY']
            assert len(entries)==2
            assert {o['body']['stop'] for o in entries}=={19990.,20010.}
            assert {o['body']['kind'] for o in entries}=={'STP LMT'}
            assert {o['body']['tif'] for o in entries}=={'GTD'}
            assert all(o['body']['good_till']==DAY.replace('-','')+' 11:30:00 America/New_York' for o in entries)
            assert all(o['body']['oca_type']==1 for o in entries)
            assert len({o['body']['oca'] for o in entries})==1
            assert not store.fill_ids() and service.states['NQ'].phase=='RESTING'
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('side',[1,-1])
def test_resting_fill_cancels_entries_and_protects_actual_size(config,tmp_path,side):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ'];price=20000+side*10
            now[0]=at('09:30:01')
            b.update_quote('NQ',price-.25,price,now[0])
            assert not s.qty # A quote crossing alone is not a Last trigger.
            b.update_trade('NQ',price,now[0])
            await service.drain()
            assert not service.halted and s.phase=='OPEN' and s.side==side and s.qty>0
            assert not service.working_entries(s)
            orders=store.orders()
            assert orders[s.stop_order]['body']['qty']==s.qty
            assert orders[s.time_order]['body']['qty']==s.qty
            assert orders[s.stop_order]['body']['side']==-side
            assert s.stop==snap(s.entry-side*10,.25,side==-1)
            assert orders[s.time_order]['body']['good_after'].endswith('15:55:00 America/New_York')
            await service.watchdog()
            assert not service.halted
        finally:store.close()
    from open_breakout.strategy import snap
    asyncio.run(run())


def test_resting_gap_leaves_triggered_limit_working_until_fill(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ'];entries=s.entry_orders.copy()
            now[0]=at('09:30:01')
            b.update_quote('NQ',19988.,19988.25,now[0])
            b.update_trade('NQ',19988.,now[0])
            await service.drain()
            assert s.qty==0 and s.attempts==1 and s.phase=='RESTING'
            assert all(b.status(oid)=='Submitted' for oid in entries)
            now[0]=at('09:30:02')
            b.update_quote('NQ',19989.5,19989.75,now[0])
            await service.drain()
            assert s.qty>0 and s.side==-1 and s.attempts==1 and not service.halted
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('reason',['cutoff','halt','shutdown','external_cancel'])
def test_resting_entries_cancel_without_fills(config,tmp_path,reason):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ'];entries=s.entry_orders.copy()
            if reason=='cutoff':
                now[0]=at('11:30:00');await service.watchdog()
            elif reason=='halt':service.halt('TEST_FEED_FAILURE')
            elif reason=='external_cancel':b.cancel(entries[0])
            else:await service.cancel_resting_entries()
            await service.drain()
            assert all(b.status(oid)=='Cancelled' for oid in entries)
            assert not store.fill_ids() and not s.qty
            if reason in {'halt','external_cancel'}:assert service.halted
            if reason=='cutoff':
                await tick(service,b,now,20010,'11:30:01')
                assert s.entry_orders==entries and not service.halted
        finally:store.close()
    asyncio.run(run())


def test_resting_partial_fill_cancels_remainder_and_tracks_burst(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ']
            oid=next(oid for oid in s.entry_orders if b.orders[oid]['body']['side']==1)
            cid=config.markets[0].execution.con_id
            for i,price in enumerate([20010.,20010.25]):
                b.record(cid,1,b.orders[oid]['body']['ref'],f'PART-{i}')
                service.fill(oid,f'PART-{i}',1,price)
            await service.drain()
            assert s.qty==2 and s.entry==20010.125 and not service.halted
            assert b.status(oid)=='Cancelled' and not service.working_entries(s)
            assert b.orders[s.stop_order]['body']['qty']==2
            assert b.orders[s.time_order]['body']['qty']==2
            service.halt('FEED_LOSS');await service.drain()
            assert b.status(s.stop_order)=='Submitted' and b.status(s.time_order)=='Submitted'
        finally:store.close()
    asyncio.run(run())


def test_resting_short_gate_risk_and_margin_caps(config,tmp_path):
    async def run():
        c=replace(config,entry_order_type='stop_limit')
        service,b,store,now=setup_service(c,tmp_path)
        service.margin_day_cap={'NQ':2}
        service.states['NQ'].score=19.99
        try:
            await tick(service,b,now,20000,'09:30:00')
            entries=[o['body'] for o in store.orders().values() if o['role']=='ENTRY']
            assert len(entries)==1 and entries[0]['side']==1 and entries[0]['qty']==2
            nq_risk=service.states['NQ'].planned_risk
            await tick(service,b,now,5000,'09:30:00',market='ES')
            total=sum(s.planned_risk for s in service.states.values())
            assert total<=c.shadow_equity*c.max_open_risk_bps/10000
            assert store.get('daily_reserved')==pytest.approx(total) and nq_risk>0
            assert not service.halted
        finally:store.close()
    asyncio.run(run())


def test_resting_reentry_requires_return_inside_and_max_three_cycles(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ']
            for attempt in range(1,4):
                now[0]=at(f'09:30:{attempt*3:02}')
                b.update_quote('NQ',20009.75,20010.,now[0]);b.update_trade('NQ',20010.,now[0])
                await service.drain()
                assert s.qty and s.attempts==attempt
                await tick(service,b,now,20000,f'09:30:{attempt*3+1:02}')
                await service.watchdog()
                await tick(service,b,now,20000,f'09:30:{attempt*3+2:02}')
            assert not s.qty and s.attempts==3 and not service.working_entries(s)
            assert not service.halted
        finally:store.close()
    asyncio.run(run())


def test_resting_cancel_uncertainty_halts(config,tmp_path):
    class IgnoreCancel(SimBroker):
        def cancel(self,oid):pass
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path,IgnoreCancel)
        service.cancel_timeout=0.
        try:
            await tick(service,b,now,20000,'09:30:00')
            now[0]=at('11:30:00');await service.watchdog();await service.drain()
            assert service.halted and 'ENTRY_CANCEL_UNCONFIRMED' in store.get('halt_reason')
            assert not store.fill_ids()
        finally:store.close()
    asyncio.run(run())


def test_resting_ibkr_order_fields_and_unhealthy_connection(config):
    from open_breakout.ibkr import IBKR
    c=replace(config,mode='paper',entry_order_type='stop_limit')
    async def make_adapter():return IBKR(c)
    adapter=asyncio.run(make_adapter());m=c.markets[0]
    body=dict(kind='STP LMT',side=-1,qty=5,stop=30688.5,limit=30688.,tif='GTD',
              good_till='20261001 11:30:00 America/New_York',oca='ENTRY',oca_type=1,
              trigger_method=2,account=c.account,ref='MNQ|SELL|OpenBreakout|2026-10-01|NQ-1-ENTRY')
    order=adapter._order(21,body)
    assert order.orderType=='STP LMT' and order.auxPrice==30688.5 and order.lmtPrice==30688.
    assert order.tif=='GTD' and order.goodTillDate==body['good_till']
    assert order.ocaType==1 and order.triggerMethod==2 and order.transmit
    adapter.healthy=False
    with pytest.raises(RuntimeError,match='Unhealthy broker connection'):adapter.send(21,m,body)


@pytest.mark.parametrize('bad',[None,'market',{},[]])
def test_resting_config_rejects_unknown_types(config,tmp_path,bad):
    raw=json.loads((tmp_path/'config.json').read_text())
    raw['entry_order_type']=bad
    path=tmp_path/'bad-entry.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='entry_order_type'):Config.load(path)


def test_resting_oca_cancel_can_precede_execution_report(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ'];buy,sell=s.entry_orders
            b.cancel(sell) # Broker cancels the sibling before delivering the fill.
            b.record(config.markets[0].execution.con_id,1,b.orders[buy]['body']['ref'],'RACE')
            service.fill(buy,'RACE',1,20010.)
            await service.drain()
            assert not service.halted and s.qty==1 and s.stop_order and s.time_order
        finally:store.close()
    asyncio.run(run())


def test_resting_broker_expiry_and_late_fill_are_safe(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ'];buy=s.entry_orders[0]
            now[0]=at('11:30:00')
            b.update_trade('NQ',20010.,now[0]) # GTD expires without a process watchdog.
            assert not s.qty and all(b.status(o)=='Cancelled' for o in s.entry_orders)
            b.record(config.markets[0].execution.con_id,1,b.orders[buy]['body']['ref'],'LATE')
            service.fill(buy,'LATE',1,20010.)
            await service.drain()
            assert service.halted and s.qty==1 and s.stop_order and s.time_order
            assert 'LATE_ENTRY_EXECUTION' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())


def test_resting_watchdog_detects_modified_entry(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            s=service.states['NQ']
            b.orders[s.entry_orders[0]]['body']['limit']+=1.
            for _ in range(3):await service.watchdog()
            await service.drain()
            assert service.halted and 'Broker resting entry differs' in store.get('halt_reason')
            assert not service.working_entries(s)
        finally:store.close()
    asyncio.run(run())


def test_config_boundaries(config,tmp_path,monkeypatch):
    config.authorize()
    paper=replace(config,mode='paper');paper.authorize()
    with pytest.raises(PermissionError):replace(paper,account='U_REAL').authorize()
    with pytest.raises(PermissionError):replace(paper,port=7496).authorize()
    live=replace(config,mode='live',account='U_REAL',allow_live=True)
    with pytest.raises(PermissionError):live.authorize(DAY)
    # The pilot block is optional since 2026-09-29; a market cap above the hard ceiling is never live-authorizable.
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} U_REAL')
    assert live.authorize(DAY)==f'LIVE {DAY} U_REAL'
    assert replace(live,markets=(replace(live.markets[0],max_contracts=60),live.markets[1])).authorize(DAY)
    over=replace(live,markets=(replace(live.markets[0],max_contracts=61),live.markets[1]))
    with pytest.raises(PermissionError,match='LIVE_HARD_MAX_CONTRACTS'):over.authorize(DAY)
    raw=json.loads((ROOT/'config/open_breakout.example.json').read_text(encoding='utf-8-sig'))
    raw['port']='7497'
    p=tmp_path/'bad.json';p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):Config.load(p)

@pytest.mark.parametrize('score,expected',[(19.999,None),(20,-1),(80,-1)])
def test_risk_threshold(score,expected):
    s=State(DAY,'NQ',40,score)
    assert on_price(s,at('09:30:00'),100,.25,at('09:30:00')) is None
    assert on_price(s,at('09:30:01'),90,.25,at('09:30:01'))==expected

@pytest.mark.parametrize('stamp',['09:30:03','09:31:00'])
def test_missed_open_blocks(stamp):
    s=State(DAY,'NQ',40,25)
    assert on_price(s,at(stamp),100,.25,at(stamp)) is None
    assert s.phase=='BLOCKED'

def test_cutoff_and_stale():
    s=State(DAY,'NQ',40,25)
    on_price(s,at('09:30:00'),100,.25,at('09:30:00'))
    assert on_price(s,at('09:30:01'),110,.25,at('09:30:06')) is None
    assert on_price(s,at('11:30:00'),110,.25,at('11:30:00')) is None

def test_sizing_costs_rounding(config):
    s=State(DAY,'NQ',40.1,25)
    p=size_order(s,config.markets[0],1,20000,20000.25,100000,config)
    assert p['qty']==6
    assert p['risk']<=150
    assert p['limit']==20000.75
    with pytest.raises(ValueError):size_order(s,config.markets[0],1,float('nan'),101,100000,config)
    with pytest.raises(ValueError):size_order(s,config.markets[0],1,90,110,100000,config)

def test_calendar_holidays_and_early_closes():
    assert calendar_dates(DAY)==('2026-09-23','2026-09-22')
    for day in ['2026-09-26','2026-11-27','2026-09-08']:
        with pytest.raises(ValueError):calendar_dates(day)

def test_risk_ignores_future_and_new_score():
    ix=pd.bdate_range('2026-09-01','2026-09-24')
    f=pd.DataFrame({'63d':30.,'main_score':90.},index=ix)
    f.loc['2026-09-24','63d']=100
    assert legacy_score(f,'2026-09-23')==30
    with pytest.raises(ValueError):legacy_score(f.loc[:'2026-09-22'],'2026-09-23')

def minutes(day):
    ix=pd.date_range(pd.Timestamp(day,tz=NY)-pd.Timedelta(days=1)+pd.Timedelta(hours=18),
        pd.Timestamp(day,tz=NY)+pd.Timedelta(hours=16,minutes=59),freq='min')
    return pd.DataFrame({'open':100.,'high':120.,'low':80.,'close':110.},index=ix)

def test_history_completeness_and_manifest(config,tmp_path):
    f=minutes('2026-09-23')
    assert session_range(f,'2026-09-23')==(120,80,110)
    with pytest.raises(ValueError):session_range(f.iloc[1:],'2026-09-23')
    risk=pd.DataFrame({'63d':25.},index=pd.bdate_range('2026-09-01','2026-09-23'))
    bars={m.name:pd.concat([minutes('2026-09-22'),f]) for m in config.markets}
    b=build_manifest(config,DAY,risk,bars,now=at('09:00:00'),roll_verified=True)
    assert b['markets']['NQ']['prior_tr']==40
    p=tmp_path/'inputs.json';p.write_text(json.dumps(b));assert load_manifest(p,config)==b
    b['score']=90;p.write_text(json.dumps(b))
    with pytest.raises(ValueError):load_manifest(p,config)
    with pytest.raises(ValueError):build_manifest(config,DAY,risk,bars,now=at('09:00:00'))

def test_stop_no_breakeven_and_time_exit(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            await tick(service,b,now,20010,'09:30:01')
            s=service.states['NQ'];assert s.phase=='OPEN';assert s.qty==6
            original=s.stop;assert original==20000
            await tick(service,b,now,20035,'09:30:02')
            assert s.stop==original # +2R does not activate break-even
            await tick(service,b,now,20010,'09:30:03')
            assert s.qty==6
            now[0]=at('15:55:00');b.update_quote('NQ',20009.75,20010,now[0])
            await service.watchdog()
            assert s.qty==0 and s.phase=='FLAT'
            assert not service.halted
            assert b.status(s.stop_order)=='Cancelled'
        finally:store.close()
    asyncio.run(run())

def test_reentry_needs_recross_and_three_attempt_limit(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            for k in range(3):
                await tick(service,b,now,20010,f'09:30:{1+k*2:02}')
                assert service.states['NQ'].attempts==k+1
                await tick(service,b,now,19999,f'09:30:{2+k*2:02}')
                await service.watchdog()
            await tick(service,b,now,20010,'09:30:07')
            assert service.states['NQ'].attempts==3 and service.states['NQ'].qty==0
        finally:store.close()
    asyncio.run(run())

def test_duplicate_execution_is_idempotent(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            s=service.states['NQ'];before=s.record();count=len(b.orders)
            service.fill(s.entry_order,'SIM-1',s.qty,20010)
            assert s.record()==before and len(b.orders)==count and not service.halted
        finally:store.close()
    asyncio.run(run())

class RejectStop(SimBroker):
    def send(self,oid,m,body):
        if body['kind']=='STP':raise RuntimeError('Stop rejected')
        super().send(oid,m,body)

def test_rejected_stop_halts_no_second_entry(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,RejectStop)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            assert service.halted
            assert len([o for o in b.orders.values() if o['body']['kind']=='LMT'])==1
            assert any(o['role']=='STOP' and o['status']=='PREPARED' for o in store.orders().values())
            await tick(service,b,now,20011,'09:30:02')
            assert service.states['NQ'].attempts==1
        finally:store.close()
    asyncio.run(run())

class PartialBroker(SimBroker):
    def execute(self,oid,qty,price):
        o=self.orders[oid]
        if o['body']['kind']!='LMT':return super().execute(oid,qty,price)
        cid=o['market'].execution.con_id
        for i,q in enumerate([2,qty-2]):
            self.record(cid,o['body']['side']*q,o['body']['ref'],f'PART-{i}')
            self.fill_callback(oid,f'PART-{i}',q,price+i*.25)
            stops=[x for x in self.orders.values() if x['body']['kind']=='STP']
            assert len(stops)==1 and stops[0]['body']['qty']==self.positions[cid]
        o['status']='Filled';self.status_callback(oid,'Filled')

def test_each_partial_is_protected_and_stop_tracks_average(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,PartialBroker)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            s=service.states['NQ']
            assert s.qty==6 and s.stop==20000 and s.phase=='OPEN'
            assert len([o for o in store.orders().values() if o['role']=='STOP'])==1
        finally:store.close()
    asyncio.run(run())

# ---- Partial fills at size (normal live sizing from 2026-09-29) ----

class IOCPartial(SimBroker):
    """The IOC entry executes `parts` (qty, price offset) and the rest is cancelled, as IB reports a partial IOC."""
    parts=[(4,0.)]
    def send(self,oid,market,body):
        if body['kind']!='LMT':return super().send(oid,market,body)
        self.orders[oid]=dict(body=body.copy(),market=market,status='Submitted')
        self.status_callback(oid,'Submitted')
        bid,ask,_=self.quote(market.name)
        for i,(q,off) in enumerate(self.parts):
            self.record(market.execution.con_id,body['side']*q,body['ref'],f'IOC-{oid}-{i}')
            self.fill_callback(oid,f'IOC-{oid}-{i}',q,(ask if body['side']==1 else bid)+off)
        self.orders[oid]['status']='Cancelled';self.status_callback(oid,'Cancelled')

def events(store):
    return [dict(kind=k,body=json.loads(b)) for k,b in store.db.execute('SELECT kind,body FROM events ORDER BY seq')]

def partial_service(config,tmp_path,parts):
    broker=type('Parts',(IOCPartial,),{'parts':parts})
    return setup_service(config,tmp_path,broker)

def test_ioc_partial_fill_protects_filled_qty_only(config,tmp_path):
    async def run():
        service,b,store,now=partial_service(config,tmp_path,[(4,0.)])
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];orders=store.orders()
            entry=[o for o in orders.values() if o['role']=='ENTRY']
            assert len(entry)==1 and entry[0]['body']['qty']==6 # sized 6, IOC filled 4
            assert s.qty==4 and s.phase=='OPEN' and s.attempts==1 and not service.halted
            assert orders[s.stop_order]['body']['qty']==4 and b.orders[s.stop_order]['body']['qty']==4
            assert orders[s.time_order]['body']['qty']==4 and b.orders[s.time_order]['body']['qty']==4
            assert s.stop==20000 and s.entry==20010
            # Risk reserved for the full order stays reserved (conservative).
            assert s.planned_risk==store.get('daily_reserved')>0
            for _ in range(4):await service.watchdog()
            assert not service.halted and service.mismatches==0
            # The 15:55 exit closes exactly the 4 held; the stop is cancelled and the journal goes flat.
            now[0]=at('15:55:00');b.update_quote('NQ',20009.75,20010,now[0]);await service.drain()
            await service.watchdog()
            assert s.qty==0 and s.phase=='FLAT' and b.positions[config.markets[0].execution.con_id]==0
            assert b.status(s.stop_order)=='Cancelled' and not service.halted
        finally:store.close()
    asyncio.run(run())

def test_second_partial_updates_average_and_resizes_same_stop(config,tmp_path):
    async def run():
        service,b,store,now=partial_service(config,tmp_path,[(2,0.),(3,1.)])
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];orders=store.orders()
            stops=[k for k,o in orders.items() if o['role']=='STOP']
            # One stop order id, modified in place for the second partial.
            assert stops==[s.stop_order] and s.qty==5
            assert s.entry==pytest.approx((2*20010+3*20011)/5)
            # Average 20010.6 - 10 = 20000.6, rounded outward (down for a long) to 20000.5.
            assert s.stop==20000.5 and b.orders[s.stop_order]['body']['stop']==20000.5
            assert b.orders[s.stop_order]['body']['qty']==5 and orders[s.time_order]['body']['qty']==5
            modified=[e for e in events(store) if e['kind']=='MODIFY_INTENT']
            assert len(modified)==1 and modified[0]['body']['id']==s.stop_order and modified[0]['body']['body']['qty']==5
            for _ in range(4):await service.watchdog()
            assert not service.halted
        finally:store.close()
    asyncio.run(run())

def test_zero_fill_at_size_consumes_attempt_and_reserves_risk(config,tmp_path):
    async def run():
        service,b,store,now=partial_service(config,tmp_path,[])
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            assert s.qty==0 and s.phase=='FLAT' and s.attempts==1 and not s.stop_order and not s.time_order
            assert store.get('daily_reserved')>0 and s.planned_risk==0 and not service.halted
        finally:store.close()
    asyncio.run(run())

class ModifyRejected(IOCPartial):
    """The second partial's in-place stop modify is rejected; ib_insync marks the trade Cancelled with the code."""
    parts=[(2,0.),(3,1.)];code=201
    def send(self,oid,market,body):
        if body['kind']=='STP' and oid in self.orders:
            self.error_codes[oid]=self.code
            self.order_error_callback(oid,self.code,'modify rejected')
            self.orders[oid]['status']='Cancelled';self.status_callback(oid,'Cancelled')
            return
        return super().send(oid,market,body)

@pytest.mark.parametrize('code',[201,10326])
def test_rejected_stop_modify_on_second_partial_halts_and_alerts(config,tmp_path,code):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,type('M',(ModifyRejected,),{'code':code}))
        service.flatten_delay=0;sent=collect(service)
        try:
            await open_nq(service,b,now);await service.drain()
            s=service.states['NQ']
            assert service.halted and any('HALTED' in x for x in sent)
            modified=[e for e in events(store) if e['kind']=='MODIFY_INTENT']
            assert len(modified)==1 and modified[0]['body']['body']['qty']==5
            flat=flatten_orders(b)
            if code==201:
                # Explicit rejection: exits cancelled, then one flatten sized to the whole journal position.
                assert len(flat)==1 and flat[0][1]['body']['qty']==5 and s.qty==0
                assert any('EMERGENCY FLATTEN SENT' in x for x in sent)
            else:
                # No explicit rejection code: no automatic flatten; halted with a loud hand-check alert.
                assert not flat and s.qty==5
                assert any('PROTECTIVE STOP CANCELLED' in x for x in sent)
        finally:store.close()
    asyncio.run(run())

def test_entry_execution_after_timed_exit_resizes_it_and_halts(config,tmp_path):
    async def run():
        service,b,store,now=partial_service(config,tmp_path,[(4,0.)])
        cid=config.markets[0].execution.con_id
        sent=collect(service)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];stop,timed=s.stop_order,s.time_order
            assert s.qty==4 and b.orders[timed]['body']['qty']==4
            # A late report of 2 more on the same IOC (re-delivered by reqExecutions).
            b.record(cid,2,b.orders[s.entry_order]['body']['ref'],'LATE-1')
            service.fill(s.entry_order,'LATE-1',2,20010.)
            assert s.qty==6 and s.stop_order==stop and s.time_order==timed
            assert b.orders[stop]['body']['qty']==6 and b.orders[timed]['body']['qty']==6
            assert b.orders[timed]['body']['ref']==store.orders()[timed]['body']['ref']
            assert service.halted and 'ENTRY_EXECUTION_AFTER_TIMED_EXIT' in store.get('halt_reason')
            await service.drain()
            for _ in range(4):await service.watchdog()
            assert service.mismatches==0
            now[0]=at('15:55:00');b.update_quote('NQ',20009.75,20010,now[0]);await service.drain()
            await service.watchdog()
            assert s.qty==0 and b.positions[cid]==0
        finally:store.close()
    asyncio.run(run())

def test_timed_exit_larger_than_position_halts(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];stop,timed=s.stop_order,s.time_order
            # Stop fills 2 of 6 but the timed exit is NOT reduced (OCA reduce missing): it would reverse.
            b.record(cid,-2,b.orders[stop]['body']['ref'],'STOP-P1')
            b.orders[stop]['body']['qty']=4
            service.fill(stop,'STOP-P1',2,19999.75)
            for _ in range(3):await service.watchdog()
            assert service.halted and 'Broker timed exit exceeds journal position' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

def test_emergency_flatten_after_partial_sizes_to_journal(config,tmp_path):
    async def run():
        service,b,store,now=partial_service(config,tmp_path,[(4,0.)])
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            b.order_error_callback(s.stop_order,201,'Order rejected')
            await service.drain()
            flat=flatten_orders(b)
            assert len(flat)==1 and flat[0][1]['body']['qty']==4 and flat[0][1]['body']['side']==-1
            assert s.qty==0 and b.positions[config.markets[0].execution.con_id]==0
        finally:store.close()
    asyncio.run(run())

def test_partial_stop_fill_oca_reduce_keeps_position_protected(config,tmp_path):
    """OCA type 2: a partial stop fill reduces the timed exit by the same amount at IB. The journal reduces,
    keeps the entry average, cancels nothing, and reconciles the reduced stop remaining without a halt."""
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];stop,timed=s.stop_order,s.time_order
            assert s.qty==6
            # IB: the stop fills 2 of 6 (remaining 4) and OCA type 2 reduces the timed exit to 4.
            b.record(cid,-2,b.orders[stop]['body']['ref'],'STOP-P1')
            b.orders[stop]['body']['qty']=4;b.orders[timed]['body']['qty']=4
            service.fill(stop,'STOP-P1',2,19999.75)
            assert s.qty==4 and s.phase=='OPEN' and s.entry==20010 and s.stop==20000
            assert b.status(stop)=='Submitted' and b.status(timed)=='Submitted' and not service.halted
            for _ in range(4):await service.watchdog()
            assert not service.halted and service.mismatches==0
            # The rest of the stop fills: flat, timed exit cancelled, no halt.
            b.execute(stop,4,19999.75)
            await service.watchdog()
            assert s.qty==0 and s.phase=='FLAT' and b.status(timed)=='Cancelled' and not service.halted
            assert b.positions[cid]==0
        finally:store.close()
    asyncio.run(run())

def test_partial_timed_exit_open_at_1556_halts(config,tmp_path):
    async def run():
        service,b,store,now=partial_service(config,tmp_path,[(4,0.)])
        cid=config.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];timed=s.time_order
            # Only 3 of the 4 fill on the 15:55 exit; one contract is still held at 15:56.
            now[0]=at('15:55:01')
            b.record(cid,-3,b.orders[timed]['body']['ref'],'TIME-P1')
            b.orders[timed]['body']['qty']=1;b.orders[s.stop_order]['body']['qty']=1
            service.fill(timed,'TIME-P1',3,20010.)
            assert s.qty==1 and not service.halted
            now[0]=at('15:56:30')
            await service.watchdog()
            assert service.halted and 'Timed exit not complete' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

def test_restart_blocks_resubmission(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
        before=len(store.orders());store.close()
        service2,b2,store2,now2=setup_service(config,tmp_path)
        try:
            assert service2.halted and len(store2.orders())==before
            assert not b2.orders
            assert service2.states['NQ'].qty==6
        finally:store2.close()
    asyncio.run(run())

def test_lock_excludes_second_process_and_foreign_position_is_ignored(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            with pytest.raises(RuntimeError):Store(tmp_path/'session.sqlite',config.fingerprint,DAY)
            # Another strategy's position is not OpenBreakout's business (orderRef ownership, 2026-09-27).
            b.foreign_positions[config.markets[0].execution.con_id]=1
            for _ in range(4):await service.watchdog()
            assert not service.halted and service.mismatches==0
        finally:store.close()
    asyncio.run(run())

def test_stream_gap_halts(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            # Gap tolerance is watchdog_stale_seconds (30s), not the 3s entry freshness.
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20000.5,'09:30:10')
            assert not service.halted
            await tick(service,b,now,20010,'09:30:41')
            assert service.halted and not b.orders
        finally:store.close()
    asyncio.run(run())

def test_ibkr_shadow_and_live_cannot_transmit(config):
    from open_breakout.ibkr import IBKR
    # Instantiate inside a running loop: ib_insync binds loop state.
    async def run():
        pytest.importorskip('ib_insync')
        ib=IBKR(config)
        with pytest.raises(PermissionError):ib.send(1,config.markets[0],{'kind':'LMT'})
        live=replace(config,mode='live')
        with pytest.raises(PermissionError):await IBKR(live).connect()
    asyncio.run(run())

class UncertainEntry(SimBroker):
    async def wait_terminal(self,oid,timeout):raise TimeoutError('Lost acknowledgement')

class NoFill(SimBroker):
    def send(self,oid,m,body):
        if body['kind']!='LMT':return super().send(oid,m,body)
        self.orders[oid]=dict(body=body,market=m,status='Cancelled')
        self.status_callback(oid,'Cancelled')

@pytest.mark.parametrize('broker_class',[UncertainEntry,NoFill])
def test_uncertain_and_zero_fill_do_not_resend(config,tmp_path,broker_class):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,broker_class)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            s=service.states['NQ']
            assert s.attempts==1 and store.get('daily_reserved')>0
            assert len([o for o in store.orders().values() if o['role']=='ENTRY'])==1
            if broker_class==UncertainEntry:
                assert service.halted and s.stop_order and s.qty>0
            else:
                assert not service.halted and s.qty==0 and s.phase=='FLAT'
            await tick(service,b,now,20011,'09:30:02')
            assert s.attempts==1 # no fresh crossing and no retry of unknown intent
        finally:store.close()
    asyncio.run(run())

def test_daily_and_pooled_open_caps(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            store.set('daily_reserved',590.)
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            assert not b.orders and not service.halted
            assert [e['body']['market'] for e in events(store) if e['kind']=='SKIP_RISK_CAP']==['NQ']
            # Open cap $200 with ES holding $190: room $10 is below one MNQ ($23.70): refused.
            store.set('daily_reserved',0.)
            service.states['ES'].planned_risk=190.
            await tick(service,b,now,20000,'09:30:02');await tick(service,b,now,20010,'09:30:03')
            assert not b.orders and service.states['NQ'].attempts==0
            # ES holding $100: room $100 fits 4 of the 6 planned (same $23.70 per-contract risk as sizing).
            service.states['ES'].planned_risk=100.
            await tick(service,b,now,20000,'09:30:04');await tick(service,b,now,20010,'09:30:05')
            entry=[o for o in b.orders.values() if o['body']['kind']=='LMT']
            assert len(entry)==1 and entry[0]['body']['qty']==4
            s=service.states['NQ']
            assert s.qty==4 and s.planned_risk==pytest.approx(4*23.7) and store.get('daily_reserved')==pytest.approx(4*23.7)
            down=[e['body'] for e in events(store) if e['kind']=='SIZED_DOWN_RISK_CAP']
            assert len(down)==1 and down[0]['planned']==6 and down[0]['qty']==4
        finally:store.close()
    asyncio.run(run())

def test_two_markets_enter_at_size_within_open_and_daily_caps(tmp_path,monkeypatch):
    """2026-09-29 live shape: MNQ 3 and MES 7 open together inside the 25bp open cap; no size-down."""
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    live=load_raw(tmp_path,normal_raw())
    async def run():
        now=[at('09:30:00')]
        b=SimBroker(live,lambda:now[0])
        store=Store(tmp_path/'s.sqlite',live.fingerprint,DAY)
        m=manifest(live)
        m['markets']['NQ']['prior_tr']=565.;m['markets']['ES']['prior_tr']=79.75
        m.pop('hash');m['hash']=digest(m)
        service=Service(live,m,store,b,lambda:now[0])
        try:
            for market,base in [('NQ',25000),('ES',6000)]:
                await tick(service,b,now,base,'09:30:00',market)
            for market,base in [('NQ',25000),('ES',6000)]:
                await tick(service,b,now,base+200 if market=='NQ' else base+30,'09:30:01',market)
            entries={o['market'].name:o['body']['qty'] for o in b.orders.values() if o['body']['kind']=='LMT'}
            assert entries=={'NQ':3,'ES':7}
            opened=sum(s.planned_risk for s in service.states.values())
            assert opened==pytest.approx(3*286.2+7*106.7) and opened<=live.shadow_equity*25/10000
            assert not [e for e in events(store) if e['kind'] in {'SIZED_DOWN_RISK_CAP','SKIP_RISK_CAP'}]
            assert not service.halted
        finally:store.close()
    asyncio.run(run())

def test_short_stop_direction(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,19990,'09:30:01')
            s=service.states['NQ'];assert s.side==-1 and s.stop==19999.75
            await tick(service,b,now,20000,'09:30:02');await service.watchdog()
            assert s.qty==0 and s.phase=='FLAT' and not service.halted
        finally:store.close()
    asyncio.run(run())

def test_stale_execution_quote_prevents_send(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00')
            b.quotes['NQ']=(20009.75,20010,at('09:29:55'))
            now[0]=at('09:30:01');service.tick('NQ',now[0],20010);await service.drain()
            # A stale quote skips this signal only: no halt, no order, no attempt consumed.
            s=service.states['NQ']
            assert not service.halted and not b.orders and s.attempts==0 and s.phase=='FLAT'
            assert store.get('daily_reserved',0.)==0
        finally:store.close()
    asyncio.run(run())

def test_openbreakout_order_from_another_client_blocks_and_foreign_ref_does_not(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            order=dict(id=1,client_id=999,con_id=cid,remaining=1,kind='LMT',side=-1,stop=0)
            b.foreign_orders=[{**order,'ref':'OTHER'},{**order,'id':2,'ref':f'MNQ|SELL|Legend_EMA|{DAY}|TARGET'}]
            await service.watchdog();assert not service.halted
            b.foreign_orders=[{**order,'ref':f'MNQ|SELL|OpenBreakout|{DAY}|NQ-1-STOP'}]
            await service.watchdog();assert service.halted and 'Unowned OpenBreakout order' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

def test_ibkr_paper_order_fields_and_live_release(config,monkeypatch):
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        paper=replace(config,mode='paper');adapter=IBKR(paper)
        m=paper.markets[0];adapter.contracts[m.execution.con_id]=object()
        sent=[]
        adapter.ib.placeOrder=lambda c,o:sent.append(o) or object()
        body=dict(kind='STP',side=-1,qty=2,stop=20000,tif='GTC',oca='test-group',account=paper.account,ref='OB:test:STOP')
        adapter.send(17,m,body)
        assert sent[0].auxPrice==20000 and sent[0].ocaType==2 and sent[0].account==paper.account
        assert sent[0].orderId==17 and sent[0].transmit
        body=dict(kind='MKT',side=-1,qty=2,tif='GTC',oca='test-group',account=paper.account,ref='OB:test:TIME',good_after='20260924 15:55:00 America/New_York')
        adapter.send(18,m,body)
        assert sent[1].goodAfterTime.endswith('America/New_York') and sent[1].ocaGroup==sent[0].ocaGroup
        live=replace(config,mode='live',account='U_REAL',allow_live=True)
        live=replace(live,markets=(replace(live.markets[0],max_contracts=61),live.markets[1]))
        monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {live.account}')
        with pytest.raises(PermissionError,match='LIVE_HARD_MAX_CONTRACTS'):await IBKR(live,session=DAY).connect()
    asyncio.run(run())

@pytest.mark.parametrize('compressed',[False,True])
def test_cli_replay_and_readonly_status(config,tmp_path,compressed):
    import subprocess
    import sys
    p=tmp_path/'inputs.json';p.write_text(json.dumps(manifest(config)))
    events=[]
    for second,price in [(0,20000),(1,20010),(2,19999)]:
        ts=at(f'09:30:{second:02}').isoformat()
        events.extend([dict(kind='quote',market='NQ',time=ts,bid=price-.25,ask=price),dict(kind='trade',market='NQ',time=ts,price=price)])
    t=tmp_path/('ticks.jsonl.gz' if compressed else 'ticks.jsonl')
    payload='\n'.join(json.dumps(e) for e in events)
    if compressed:
        import gzip
        with gzip.open(t,'wt') as f:f.write(payload)
    else:t.write_text(payload)
    state=tmp_path/'cli.sqlite'
    result=subprocess.run([sys.executable,'-m','open_breakout','replay','--config',str(tmp_path/'config.json'),
        '--inputs',str(p),'--ticks',str(t),'--state',str(state)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout);assert not output['halted']
    assert output['states'][0]['qty']==0 and output['states'][0]['attempts']==1
    result=subprocess.run([sys.executable,'-m','open_breakout','status','--state',str(state)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['meta']['daily_reserved']>0

class StopFillsDuringAck(SimBroker):
    async def wait_ack(self,oid,timeout):
        order=self.orders[oid]
        if order['body']['kind']=='STP':
            self.execute(oid,order['body']['qty'],order['body']['stop'])
        await super().wait_ack(oid,timeout)

def test_stop_fill_during_ack_does_not_create_orphan_time_exit(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,StopFillsDuringAck)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            await service.watchdog()
            assert not [o for o in b.orders.values() if o['body']['kind']=='MKT']
            assert service.states['NQ'].phase=='FLAT' and not service.halted
        finally:store.close()
    asyncio.run(run())

def test_changed_broker_stop_quantity_halts(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            stop=service.states['NQ'].stop_order
            b.orders[stop]['body']['qty']=1
            await service.watchdog();await service.watchdog();assert not service.halted
            await service.watchdog()
            assert service.halted and 'protective stop differs' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

@pytest.mark.parametrize('mode',['paper','live'])
def test_shadow_standby_refuses_order_modes(config,mode):
    from open_breakout.standby import validate_launch
    with pytest.raises(PermissionError):validate_launch(replace(config,mode=mode),DAY,at('08:00:00'))

def test_shadow_standby_bounds(config):
    from open_breakout.standby import validate_launch
    prep,opening,end=validate_launch(config,DAY,at('08:00:00'))
    assert prep.hour==9 and opening.minute==30 and end.hour==16
    with pytest.raises(ValueError):validate_launch(config,DAY,at('09:30:00'))

def test_compressed_capture_roundtrip(tmp_path):
    import gzip
    from open_breakout.standby import Capture
    capture=Capture(tmp_path/'captures')
    event=dict(kind='trade',market='NQ',time=at('09:30:00').isoformat(),price=20000)
    capture.write(event,at('09:30:01'));capture.write(event,at('10:00:00'));capture.close()
    files=sorted((tmp_path/'captures').glob('*.gz'));assert len(files)==2
    with gzip.open(files[0],'rt') as f:r=json.loads(f.readline())
    assert r['received_at']==at('09:30:01').isoformat() and r['time']==event['time']

def test_shadow_standby_full_session_simulates_only(config,tmp_path,monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    from open_breakout import standby
    from open_breakout.__main__ import main_async
    clock=[at('08:59:59')]
    class ClockDateTime(datetime):
        @classmethod
        def now(cls,tz=None):return clock[0].astimezone(tz) if tz else clock[0]
    monkeypatch.setattr(standby,'datetime',ClockDateTime)
    class Feed:
        instance=None
        def __init__(self,c):
            Feed.instance=self;self.config=c;self.healthy=True;self.connected=False;self.quotes={}
            self.ib=SimpleNamespace(client=SimpleNamespace(),isConnected=lambda:self.connected)
        async def connect(self):
            with pytest.raises(PermissionError):self.ib.client.placeOrder()
            self.connected=True
        def subscribe(self,on_tick,capture):self.on_tick=on_tick;self.capture=capture
        async def history(self):
            return {m.name:pd.concat([minutes('2026-09-22'),minutes('2026-09-23')]) for m in config.markets}
        def close(self):self.connected=False
    monkeypatch.setattr(standby,'IBKR',Feed)
    original_sleep=asyncio.sleep
    times=iter(['09:00:00','09:30:00','09:30:01','15:55:00','16:01:00'])
    async def advance(seconds):
        stamp=next(times);clock[0]=at(stamp)
        feed=Feed.instance
        if stamp>='09:30:00' and stamp<'16:01:00':
            for m,base in [('NQ',20000),('ES',5000)]:
                price=base+(10 if stamp>'09:30:00' else 0)
                e=dict(kind='quote',market=m,time=clock[0].isoformat(),bid=price-.25,ask=price)
                feed.capture(e);feed.on_tick(m,clock[0],price)
        for _ in range(20):await original_sleep(0)
    monkeypatch.setattr(standby.asyncio,'sleep',advance)
    risk=tmp_path/'risk.parquet'
    pd.DataFrame({'63d':25.},index=pd.bdate_range('2026-09-01','2026-09-23')).to_parquet(risk)
    run=tmp_path/'run'
    asyncio.run(main_async(SimpleNamespace(command='shadow-session',config=str(tmp_path/'config.json'),
        session=DAY,risk_parquet=str(risk),state_dir=str(run),roll_verified=True)))
    with sqlite3.connect(run/'runtime.sqlite') as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT * FROM meta')}
    assert meta['phase']=='SESSION_COMPLETE'
    with sqlite3.connect(run/'trades.sqlite') as db:
        fills=db.execute('SELECT COUNT(*) FROM fills').fetchone()[0]
        states=[json.loads(r[0]) for r in db.execute('SELECT body FROM states')]
    assert fills==2 and all(s['qty']==0 for s in states)
    # Amendment 2026-09-28: the session's own R is written to the runtime meta at completion and equals
    # a recomputation from the journal; no prior shadow journal for 09-23 means a null prior-day R.
    from open_breakout.store import journal_day_r
    with sqlite3.connect(run/'trades.sqlite') as db:
        recomputed=journal_day_r(db,['NQ','ES'])
    assert set(meta['day_r'])=={'NQ','ES'} and meta['day_r']==recomputed
    # Two fills: one NQ round trip (timed exit at 15:55), no ES trade.
    assert meta['day_r']['NQ']['trades']==1 and isinstance(meta['day_r']['NQ']['r'],float)
    assert meta['day_r']['ES']['r'] is None and meta['day_r']['ES']['trades']==0
    inputs=json.loads((run/'inputs.json').read_text())
    for item in inputs['markets'].values():
        assert item['prior_day_r'] is None and item['prior_big_win'] is None and item['prior_day_source'] is None
        assert 'no shadow journal for 2026-09-23' in item['prior_day_note'] and not item['skip_prior_range']


# ---- Live one-contract pilot ----

LIVE_ACCOUNT='U9990001'

def live_raw(**changes):
    raw=json.loads((ROOT/'config/open_breakout.example.json').read_text(encoding='utf-8-sig'))
    raw.update(mode='live',allow_live=True,account=LIVE_ACCOUNT,port=7496,client_id=927999,
               pilot={'max_contracts_per_market':1},entry_order_type='ioc')
    for i,m in enumerate(raw['markets']):
        m['max_contracts']=1
        for j,key in enumerate(['signal','execution']):
            m[key].update(con_id=100+i*2+j,expiry='20261218')
    raw.update(changes)
    return raw

def load_raw(tmp_path,raw,name='live.json'):
    p=tmp_path/name;p.write_text(json.dumps(raw));return Config.load(p)

@pytest.fixture
def live(tmp_path,monkeypatch):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    return load_raw(tmp_path,live_raw())

def test_live_config_requires_exact_session_ack(tmp_path,monkeypatch):
    from open_breakout.config import LIVE_HARD_MAX_CONTRACTS
    # Fat-finger ceiling only since the owner decision of 2026-09-28 late (no per-market contract cap).
    assert LIVE_HARD_MAX_CONTRACTS==60
    c=load_raw(tmp_path,live_raw())
    assert c.mode=='live' and c.pilot_max_contracts==1
    monkeypatch.delenv('OPEN_BREAKOUT_LIVE_ACK',raising=False)
    with pytest.raises(PermissionError):c.authorize(DAY)
    for wrong in [f'{LIVE_ACCOUNT}:{c.fingerprint}',f'LIVE 2026-09-25 {LIVE_ACCOUNT}',f'LIVE {DAY} U0000000',
                  f'live {DAY} {LIVE_ACCOUNT}',f'LIVE {DAY} {LIVE_ACCOUNT} ']:
        monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',wrong)
        with pytest.raises(PermissionError):c.authorize(DAY)
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    with pytest.raises(PermissionError):c.authorize()
    assert c.authorize(DAY)==f'LIVE {DAY} {LIVE_ACCOUNT}'

@pytest.mark.parametrize('mutate',[
    lambda r:r['markets'][0].update(max_contracts=61),
    lambda r:r['markets'][1].update(max_contracts=61),
    lambda r:r['markets'][0].update(max_contracts=0),
    lambda r:r['markets'][0].update(max_contracts=2.0),
    lambda r:r.update(account='DU1234567'),
    lambda r:r.update(pilot={'max_contracts_per_market':61}),
    lambda r:r.update(pilot={'max_contracts_per_market':0}),
    lambda r:r.update(pilot={'max_contracts_per_market':10.0}),
    lambda r:r.update(pilot={'max_contracts_per_market':True}),
    lambda r:r.update(pilot={'max_contracts_per_market':1,'extra':1}),
    lambda r:r.update(allow_live=False),
])
def test_live_config_structural_rejections(tmp_path,mutate):
    raw=live_raw();mutate(raw)
    with pytest.raises((PermissionError,ValueError)):load_raw(tmp_path,raw)

def normal_raw(pilot=None,caps=(60,60)):
    """Live sizing from 2026-09-29: the live config shape (no pilot block, fat-finger ceiling 60/60 since the
    owner decision of 2026-09-28 late, $750k base, 15/10 bp)."""
    raw=live_raw(shadow_equity=750000,max_daily_risk_bps=75,max_open_risk_bps=25)
    if pilot is None:raw.pop('pilot')
    else:raw['pilot']={'max_contracts_per_market':pilot}
    for m,cap,bps in zip(raw['markets'],caps,(15,10)):m.update(max_contracts=cap,risk_bps=bps)
    return raw

def test_live_effective_caps_pilot_optional_and_hard_ceiling(tmp_path):
    from open_breakout.config import LIVE_HARD_MAX_CONTRACTS
    # The live shape: no pilot block, 60/60 (the fat-finger ceiling).
    c=load_raw(tmp_path,normal_raw())
    assert [c.max_contracts_for(m) for m in c.markets]==[60,60] and c.pilot_max_contracts==0
    # Pilot block optional: the market caps rule, still under the hard ceiling.
    c=load_raw(tmp_path,normal_raw(pilot=None,caps=(20,7)),'nopilot.json')
    assert c.pilot_max_contracts==0 and [c.max_contracts_for(m) for m in c.markets]==[20,7]
    # A pilot block, when present, is accepted up to 60 and still applied.
    c=load_raw(tmp_path,normal_raw(pilot=60),'pilot60.json')
    assert c.pilot_max_contracts==60 and [c.max_contracts_for(m) for m in c.markets]==[60,60]
    c=load_raw(tmp_path,normal_raw(pilot=3),'pilot3.json')
    assert [c.max_contracts_for(m) for m in c.markets]==[3,3]
    # Hard ceiling applies even to a config object built around validation.
    over=replace(c,pilot_max_contracts=0,markets=tuple(replace(m,max_contracts=100) for m in c.markets))
    assert [over.max_contracts_for(m) for m in over.markets]==[LIVE_HARD_MAX_CONTRACTS]*2
    with pytest.raises(PermissionError):over.validate_live()
    # Shadow/paper keep their own caps (no live clamp).
    shadow=replace(c,mode='shadow')
    assert [shadow.max_contracts_for(m) for m in shadow.markets]==[60,60]

def test_repo_live_and_shadow_configs_carry_the_fat_finger_ceiling():
    """The launcher configs: no pilot block in live, 60 per market in both (shadow simulates the live rule)."""
    from open_breakout.config import LIVE_HARD_MAX_CONTRACTS
    runs=ROOT/'artifacts/open_breakout_runs'
    live_p,shadow_p=runs/'config-20260925-live.json',runs/'config-20260925-shadow.json'
    if not live_p.exists() or not shadow_p.exists():pytest.skip('launcher configs not present in this checkout')
    live_cfg=json.loads(live_p.read_text(encoding='utf-8-sig'))
    shadow_cfg=json.loads(shadow_p.read_text(encoding='utf-8-sig'))
    assert 'pilot' not in live_cfg and live_cfg['mode']=='live'
    for raw in (live_cfg,shadow_cfg):
        assert [m['max_contracts'] for m in raw['markets']]==[LIVE_HARD_MAX_CONTRACTS]*2
    assert Config.load(shadow_p).max_contracts_for(Config.load(shadow_p).markets[0])==60

def test_live_config_fingerprint_changes_with_caps_but_shadow_does_not(config,tmp_path):
    one=load_raw(tmp_path,live_raw(),'one.json')
    ten=load_raw(tmp_path,normal_raw(),'ten.json')
    assert one.fingerprint!=ten.fingerprint
    assert Config.load(tmp_path/'config.json').fingerprint==config.fingerprint

def test_shadow_fingerprint_unchanged_without_pilot(config,tmp_path):
    raw=json.loads((tmp_path/'config.json').read_text())
    assert 'pilot' not in raw and Config.load(tmp_path/'config.json').fingerprint==config.fingerprint
    assert config.pilot_max_contracts==0 and config.authorize() is None

def test_ceiling_halts_key_optional_and_boolean(config,tmp_path):
    raw=json.loads((tmp_path/'config.json').read_text())
    assert config.ceiling_halts is False and 'ceiling_halts' not in raw
    on=load_raw(tmp_path,{**raw,'ceiling_halts':True},'on.json')
    assert on.ceiling_halts is True and on.fingerprint!=config.fingerprint
    with pytest.raises(ValueError,match='ceiling_halts'):load_raw(tmp_path,{**raw,'ceiling_halts':1},'bad.json')

def test_live_sizing_clamps_to_pilot_ceiling_and_zero_stays_zero(config):
    s=State(DAY,'NQ',40.1,25)
    shadow=size_order(s,config.markets[0],1,20000,20000.25,100000,config)
    assert shadow['qty']==6 # live clamp is not applied in shadow
    # One-contract pilot ceiling (the 2026-09-28 shape) still clamps to 1.
    one=replace(config,mode='live',pilot_max_contracts=1)
    p=size_order(s,config.markets[0],1,20000,20000.25,100000,one)
    assert p['qty']==1 and p['risk']==pytest.approx(shadow['risk']/6)
    assert size_order(s,config.markets[0],1,20000,20000.25,1000,one)['qty']==0
    # Normal sizing: no clamp below the risk-based size when the ceiling is above it.
    ten=replace(config,mode='live',pilot_max_contracts=10)
    assert size_order(s,config.markets[0],1,20000,20000.25,100000,ten)['qty']==6
    assert size_order(s,config.markets[0],1,20000,20000.25,100000,replace(ten,pilot_max_contracts=4))['qty']==4
    # Risk size far above every cap: min(market cap, pilot, hard ceiling 60).
    assert size_order(s,config.markets[0],1,20000,20000.25,10**7,ten)['qty']==min(10,config.markets[0].max_contracts)
    big=replace(ten,pilot_max_contracts=0,markets=tuple(replace(m,max_contracts=100) for m in ten.markets))
    assert size_order(s,big.markets[0],1,20000,20000.25,10**8,big)['qty']==60
    # Shadow at the same market cap is not clamped by the live ceiling.
    assert size_order(s,big.markets[0],1,20000,20000.25,10**8,replace(big,mode='shadow'))['qty']==100

@pytest.mark.parametrize('side',[1,-1])
def test_live_normal_sizing_at_2026_09_29_ranges(tmp_path,side):
    """Tuesday 2026-09-29 staging numbers: NQ prior TR 565.0, ES 79.75 on the $750k basis, 15/10 bp."""
    live=load_raw(tmp_path,normal_raw())
    nq,es=live.markets
    s_nq,s_es=State(DAY,'NQ',565.,25),State(DAY,'ES',79.75,25)
    p=size_order(s_nq,nq,side,24999.75,25000.,live.shadow_equity,live)
    # MNQ: stop 141.25 pts x $2 = 282.50 + fees 2 x 0.85 + 4-tick reserve (1 pt x $2) = 286.20 per contract.
    assert p['qty']==3 and p['risk']==pytest.approx(3*286.2)
    q=size_order(s_es,es,side,5999.75,6000.,live.shadow_equity,live)
    # MES: 19.9375 rounds outward to 20.00 pts x $5 = 100 + 1.70 + 4-tick reserve (1 pt x $5) = 106.70.
    assert q['qty']==7 and q['risk']==pytest.approx(7*106.7)
    assert abs(q['limit']-snap_stop(q['limit'],side,19.9375))==20.
    # Both at once fit the 25bp open cap ($1,875); three attempts each fit the 75bp daily cap ($5,625).
    assert p['risk']+q['risk']<=live.shadow_equity*25/10000
    assert 3*(p['risk']+q['risk'])<=live.shadow_equity*75/10000

def snap_stop(limit,side,distance):
    from open_breakout.strategy import snap
    return snap(limit-side*distance,.25,side==-1)

def open_nq(service,b,now):
    async def go():
        await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
    return go()

def flatten_orders(b):
    return [(oid,o) for oid,o in b.orders.items() if o['body']['kind']=='MKT' and 'good_after' not in o['body']]

@pytest.mark.parametrize('how',['error_201','inactive','cancelled_201','cancelled_10147'])
def test_explicit_stop_rejection_flattens_once(config,tmp_path,how):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];stop,timed=s.stop_order,s.time_order
            assert s.phase=='OPEN' and s.qty==6 and timed
            if how=='error_201':b.order_error_callback(stop,201,'Order rejected')
            else:
                # Status path, as ib_insync delivers it: log code recorded before the status event.
                status='Inactive' if how=='inactive' else 'Cancelled'
                if how.startswith('cancelled'):b.error_codes[stop]=int(how.split('_')[1])
                b.orders[stop]['status']=status;b.status_callback(stop,status)
            await service.drain()
            flat=flatten_orders(b)
            assert len(flat)==1
            oid,o=flat[0]
            assert o['body']['side']==-1 and o['body']['qty']==6 and o['body']['tif']=='DAY'
            assert o['body']['account']==config.account and o['body']['ref'].split('|')[2]=='OpenBreakout'
            assert b.positions[config.markets[0].execution.con_id]==0 and s.qty==0
            assert b.status(timed)=='Cancelled' and service.halted
            assert 'PROTECTIVE_STOP_REJECTED' in store.get('halt_reason') and store.get('flattened')==['NQ']
            # Repeated rejection messages never issue a second flatten.
            b.order_error_callback(stop,201,'Order rejected');b.status_callback(stop,'Cancelled')
            await service.drain()
            assert len(flatten_orders(b))==1
            await service.watchdog();assert s.phase=='FLAT'
        finally:store.close()
    asyncio.run(run())

@pytest.mark.parametrize('code',[2104,2106,2108,2109,2158,399,404])
def test_warning_codes_never_flatten(config,tmp_path,code):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            b.order_error_callback(s.stop_order,code,'warning')
            await service.drain()
            assert not flatten_orders(b) and not service.halted and s.qty==6
            assert b.status(s.stop_order)=='Submitted'
        finally:store.close()
    asyncio.run(run())

def test_stop_cancel_after_timed_exit_fill_does_not_flatten(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            # OCA: the 15:55 exit fills and the broker cancels the sibling stop.
            now[0]=at('15:55:00');b.update_quote('NQ',20009.75,20010,now[0])
            await service.drain()
            assert not flatten_orders(b) and not service.halted and s.qty==0
        finally:store.close()
    asyncio.run(run())

@pytest.mark.parametrize('how,reason',[('account_flat_ceiling_halts','FLATTEN_SKIPPED_POSITION_MISMATCH'),
                                        ('own_executions_extra','FLATTEN_SKIPPED_POSITION_MISMATCH'),
                                        ('own_executions_missing','FLATTEN_SKIPPED_OWN_POSITION_UNKNOWN'),
                                        ('executions_request_failed','FLATTEN_SKIPPED_OWN_POSITION_UNKNOWN')])
def test_broker_position_mismatch_skips_flatten(config,tmp_path,how,reason):
    async def run():
        c=replace(config,ceiling_halts=True) if how=='account_flat_ceiling_halts' else config
        service,b,store,now=setup_service(c,tmp_path)
        service.flatten_delay=0
        cid=config.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            sent=collect(service)
            # Account ceiling (flattened by hand, no orderRef) with ceiling_halts, own executions differing from
            # the journal, or own position UNKNOWN (journaled fills absent / request failed): never flatten.
            if how=='account_flat_ceiling_halts':b.positions[cid]=0
            elif how=='own_executions_extra':b.record(cid,1,f'MNQ|BUY|OpenBreakout|{DAY}|NQ-9-ENTRY','X-1')
            elif how=='own_executions_missing':b.executions.clear()
            else:b.executions_error='TimeoutError'
            b.order_error_callback(s.stop_order,201,'Order rejected')
            await service.drain()
            assert not flatten_orders(b) and service.halted
            assert reason in store.get('halt_reason') and 'FLATTEN_SKIPPED' in kinds(store)
            assert any('FLATTEN BY HAND' in x for x in sent)
        finally:store.close()
    asyncio.run(run())

def test_account_flat_by_hand_warns_and_never_flattens(config,tmp_path):
    """Ceiling breach at flatten time: the account may no longer hold our position (hand flatten), so a
    speculative flatten could reverse it. Uncertain means no order: warn, halt, FLATTEN BY HAND."""
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            sent=collect(service)
            b.positions[config.markets[0].execution.con_id]=0
            b.order_error_callback(s.stop_order,201,'Order rejected')
            await service.drain()
            assert flatten_orders(b)==[]
            assert 'POSITION_CEILING_WARNING' in kinds(store) and 'FLATTEN_SKIPPED' in kinds(store)
            assert service.halted
            assert any('POSITION CEILING WARNING' in x for x in sent)
            assert any('FLATTEN BY HAND' in x for x in sent)
        finally:store.close()
    asyncio.run(run())

def test_uncertain_stop_ack_halts_without_flatten(config,tmp_path):
    class SilentStop(SimBroker):
        async def wait_ack(self,oid,timeout):
            if self.orders[oid]['body']['kind']=='STP':raise TimeoutError('Uncertain acknowledgement')
            await super().wait_ack(oid,timeout)
    async def run():
        service,b,store,now=setup_service(config,tmp_path,SilentStop)
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            assert service.halted and not flatten_orders(b) and service.states['NQ'].qty==6
        finally:store.close()
    asyncio.run(run())

def test_live_timed_exit_and_stop_order_fields(config,live,tmp_path):
    from daily_execution_report import parse_ref
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ'];orders=store.orders()
            stop,timed=orders[s.stop_order]['body'],orders[s.time_order]['body']
        finally:store.close()
        assert parse_ref(timed['ref'])==('OpenBreakout',DAY) and parse_ref(stop['ref'])==('OpenBreakout',DAY)
        assert timed['ref'].startswith('MNQ|BUY|') and timed['ref']!=stop['ref']
        adapter=IBKR(live,session=DAY)
        m=live.markets[0];adapter.contracts[m.execution.con_id]=object()
        sent=[]
        adapter.ib.placeOrder=lambda c,o:sent.append(o) or object()
        adapter.send(21,m,{**stop,'qty':1,'account':live.account})
        adapter.send(22,m,{**timed,'qty':1,'account':live.account})
        st,tx=sent
        assert tx.orderType=='MKT' and tx.tif=='GTC' and tx.goodAfterTime==f'{DAY.replace("-","")} 15:55:00 America/New_York'
        assert tx.ocaGroup==st.ocaGroup==s.oca and tx.ocaType==2==st.ocaType
        assert tx.account==st.account==live.account and tx.orderRef==timed['ref'] and st.orderRef==stop['ref']
        assert st.orderType=='STP' and st.tif=='GTC' and st.auxPrice==stop['stop'] and st.action=='SELL'
        with pytest.raises(PermissionError,match='quantity cap'):adapter.send(23,m,{**timed,'qty':2,'account':live.account})
        with pytest.raises(PermissionError):adapter.send(24,m,{**timed,'qty':1,'account':'U0000000'})
        assert len(sent)==2
    asyncio.run(run())

def test_live_send_requires_ack_each_time(live,monkeypatch):
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        adapter=IBKR(live,session=DAY)
        m=live.markets[0];adapter.contracts[m.execution.con_id]=object()
        sent=[]
        adapter.ib.placeOrder=lambda c,o:sent.append(o) or object()
        monkeypatch.delenv('OPEN_BREAKOUT_LIVE_ACK')
        body=dict(kind='MKT',side=-1,qty=1,tif='DAY',account=live.account,ref='x')
        with pytest.raises(PermissionError):adapter.send(1,m,body)
        with pytest.raises(PermissionError):await adapter.connect()
        with pytest.raises(PermissionError):IBKR(live).send(1,m,body)
        assert not sent
    asyncio.run(run())

def test_order_gate_blocks_until_armed():
    from types import SimpleNamespace
    from open_breakout.standby import OrderGate
    calls=[]
    client=SimpleNamespace(placeOrder=lambda *a:calls.append(a))
    gate=OrderGate(client)
    with pytest.raises(PermissionError):client.placeOrder(1,'c',SimpleNamespace(whatIf=False))
    client.placeOrder(2,'c',SimpleNamespace(whatIf=True))
    gate.open=True;client.placeOrder(3,'c',SimpleNamespace(whatIf=False))
    assert [c[0] for c in calls]==[2,3]

def test_shadow_session_refuses_live_config_and_live_session_refuses_shadow(config,live,tmp_path):
    from types import SimpleNamespace
    from open_breakout.__main__ import main_async
    from open_breakout.standby import validate_launch, validate_live_launch
    with pytest.raises(PermissionError):validate_launch(live,DAY,at('08:00:00'))
    with pytest.raises(PermissionError):validate_live_launch(config,DAY,at('08:00:00'))
    assert validate_live_launch(live,DAY,at('08:00:00'))[3]==f'LIVE {DAY} {LIVE_ACCOUNT}'
    def args(command,path):
        return SimpleNamespace(command=command,config=str(path),session=DAY,risk_parquet='missing.parquet',
                               state_dir=str(tmp_path/command),roll_verified=True)
    with pytest.raises(PermissionError):asyncio.run(main_async(args('shadow-session',tmp_path/'live.json')))
    with pytest.raises(PermissionError):asyncio.run(main_async(args('live-session',tmp_path/'config.json')))
    assert not (tmp_path/'shadow-session').exists() and not (tmp_path/'live-session').exists()

def test_live_session_cli_requires_env_ack(live,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from open_breakout.__main__ import main_async
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE 2026-09-25 {LIVE_ACCOUNT}')
    a=SimpleNamespace(command='live-session',config=str(tmp_path/'live.json'),session=DAY,risk_parquet='missing.parquet',
                      state_dir=str(tmp_path/'run'),roll_verified=True)
    with pytest.raises(PermissionError):asyncio.run(main_async(a))
    assert not (tmp_path/'run').exists()

class FakePreflight:
    """Read-only transport double for preflight: account positions, all-client open orders, own executions."""
    ack='ack';healthy=True;contract_report={'MNQ':{'ok':True}};data_types={1:1}
    def __init__(self,config,positions,orders,margin,own=None):
        from types import SimpleNamespace as NS
        self.config=config
        now=datetime.now(NY)
        self.last_seen={f'{m.name}:{k}':now for m in config.markets for k in ('trade','quote')}
        self.margin=margin;self.own=own or {}
        async def pos():return positions
        async def orders_():return orders
        self.ib=NS(reqPositionsAsync=pos,reqAllOpenOrdersAsync=orders_)
    async def own_position(self,con_id):return self.own.get(con_id,0)
    async def account_values(self):return {'NetLiquidation':600000.,'ExcessLiquidity':500000.}
    quotes={'NQ':(24999.75,25000.,None),'ES':(5999.75,6000.,None)}
    def quote(self,name):return self.quotes[name]
    async def what_if(self,m,body):
        from types import SimpleNamespace as NS
        # One contract (basic check), the planned size or the effective cap; never above the cap.
        assert 1<=body['qty']<=self.config.max_contracts_for(m) and body['kind']=='MKT'
        margin=self.margin(m,body) if callable(self.margin) else self.margin
        return NS(initMarginChange=str(margin),maintMarginChange='1',commission=0.,warningText='')

def open_trade(con_id,ref,client=0,order_type='STP',action='SELL'):
    from types import SimpleNamespace as NS
    return NS(contract=NS(conId=con_id),order=NS(clientId=client,orderId=5,permId=9,orderType=order_type,action=action,
              totalQuantity=1,orderRef=ref),orderStatus=NS(status='PreSubmitted'))

def test_preflight_scopes_to_openbreakout_refs_and_reports_other_book(live):
    from types import SimpleNamespace as NS
    from open_breakout.standby import preflight
    mnq,mes=[m.execution.con_id for m in live.markets]
    async def run():
        clean=await preflight(FakePreflight(live,[],[],3000.),wait_seconds=0)
        assert clean['ok'] and set(clean['what_if'])=={'MNQ:BUY','MNQ:SELL','MES:BUY','MES:SELL'}
        # Legend EMA holds 1 MES with its TARGET working: other_book only, preflight passes.
        pos=[NS(account=live.account,contract=NS(conId=mes),position=1.)]
        legend=[open_trade(mes,f'MES|SELL|Legend_EMA|{DAY}|TARGET',client=7,order_type='LMT'),
                open_trade(mnq,'manual')]
        shared=await preflight(FakePreflight(live,pos,legend,3000.),wait_seconds=0)
        assert shared['ok'],shared['failures']
        assert shared['own_positions']=={'MNQ':0,'MES':0} and shared['working_orders']==[]
        other=shared['other_book']
        assert other['MES']['account_position']==1 and other['MES']['other_position']==1
        assert [w['ref'] for w in other['MES']['working_orders']]==[f'MES|SELL|Legend_EMA|{DAY}|TARGET']
        assert [w['ref'] for w in other['MNQ']['working_orders']]==['manual']
        # A working OpenBreakout order from any client, or an OpenBreakout-attributed position, fails.
        ours=[open_trade(mes,f'MES|SELL|OpenBreakout|{DAY}|ES-1-STOP',client=0)]
        bad=await preflight(FakePreflight(live,pos,ours+legend,200000.,own={mnq:1}),wait_seconds=0)
        assert not bad['ok']
        text=' '.join(bad['failures'])
        assert 'WORKING_ORDER:MES:client0' in text and 'OWN_POSITION:MNQ:1' in text and 'MARGIN:MNQ:BUY' in text
        assert 'MES:1' not in text # the Legend position is never a failure
        # Unreadable executions: flat cannot be verified, so preflight fails closed (before any order).
        blind=FakePreflight(live,pos,legend,3000.)
        async def fails(con_id):raise TimeoutError('reqExecutions')
        blind.own_position=fails
        closed=await preflight(blind,wait_seconds=0)
        assert not closed['ok'] and closed['own_positions'] is None
        assert [f for f in closed['failures'] if f.startswith('OWN_POSITION')]==['OWN_POSITION_UNKNOWN:executions request failed (TimeoutError: reqExecutions)']
        assert closed['other_book']['MES']['other_position'] is None
    asyncio.run(run())

def tuesday_manifest(cfg,nq_tr=565.,es_tr=79.75,**es_fields):
    """2026-09-29 staging ranges (NQ prior TR 565.0, ES 79.75)."""
    m=manifest(cfg)
    m['markets']['NQ']['prior_tr']=nq_tr;m['markets']['ES'].update(prior_tr=es_tr,**es_fields)
    m.pop('hash');m['hash']=digest(m)
    return m

# 2026-09-28 evening what-ifs, per contract (BUY / SELL): MNQ 6,728.37 / 6,116.70; MES 3,486.38 / 2,860.58.
IB_MARGIN={('MNQ',1):6728.37,('MNQ',-1):6116.70,('MES',1):3486.375,('MES',-1):2860.578}

def margin_log(seen,per=None):
    def f(m,body):
        seen.append((m.execution.symbol,body['side'],body['qty']))
        return body['qty']*(per if per is not None else IB_MARGIN[(m.execution.symbol,body['side'])])
    return f

def test_preflight_at_connect_previews_reference_size_as_warning_only(tmp_path,monkeypatch):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    from open_breakout.standby import preflight, REFERENCE_WHATIF_CONTRACTS
    assert REFERENCE_WHATIF_CONTRACTS==10
    live=load_raw(tmp_path,normal_raw())
    assert [live.max_contracts_for(m) for m in live.markets]==[60,60]
    seen=[]
    async def run():
        # Limit: min(500k excess, 600k NLV) x 0.2 = 100k. The fat-finger cap (60) is never previewed; the
        # reference size 10/10 is: 67,283.70 + 34,863.75 > 100k, a warning only.
        r=await preflight(FakePreflight(live,[],[],margin_log(seen)),wait_seconds=0)
        assert r['ok'] and r['failures']==[] and r['what_if_basis']=='reference'
        assert r['margin_total']==pytest.approx(10*6728.37+10*3486.375) and r['margin_limit']==pytest.approx(100000.)
        assert r['warnings']==[f'MARGIN_TOTAL_AT_REFERENCE:{10*6728.37+10*3486.375:.2f}>100000.00']
        assert r['what_if_qty']=={'MNQ':{'BUY':10,'SELL':10},'MES':{'BUY':10,'SELL':10}}
        # The one-contract basic check runs as well, both sides.
        assert sorted(seen)==sorted([(s,d,q) for s in ('MNQ','MES') for d in (1,-1) for q in (1,10)])
        assert set(r['what_if_one_contract'])=={'MNQ:BUY','MNQ:SELL','MES:BUY','MES:SELL'}
        # Even a single market over the limit at the reference size is a warning, not a failure.
        big=await preflight(FakePreflight(live,[],[],margin_log([],per=20000.)),wait_seconds=0)
        assert big['ok'] and any(w.startswith('MARGIN_AT_REFERENCE:MNQ:BUY') for w in big['warnings'])
        # An effective cap below the reference size (a pilot ceiling) is what gets previewed.
        three=load_raw(tmp_path,normal_raw(pilot=3,caps=(10,8)),'three.json')
        seen.clear()
        r3=await preflight(FakePreflight(three,[],[],margin_log(seen)),wait_seconds=0)
        assert r3['what_if_qty']=={'MNQ':{'BUY':3,'SELL':3},'MES':{'BUY':3,'SELL':3}} and {q for *_,q in seen}=={1,3}
        # The one-contract check still fails the preflight (contract not tradeable / warning).
        bad=await preflight(FakePreflight(live,[],[],margin_log([],per=150000.)),wait_seconds=0)
        assert not bad['ok'] and 'MARGIN:MNQ:BUY:150000.0' in bad['failures']
    asyncio.run(run())

def test_preflight_at_arming_uses_planned_sizes_and_fails_only_above_limit(tmp_path,monkeypatch):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    from open_breakout.standby import preflight
    live=load_raw(tmp_path,normal_raw())
    m=tuesday_manifest(live)
    seen=[]
    async def run():
        # Tonight's per-contract margins: at the cap the total (102,147) would exceed 100k; planned 3/7 fits.
        r=await preflight(FakePreflight(live,[],[],margin_log(seen)),manifest=m,wait_seconds=0)
        assert r['ok'],r['failures']
        assert r['what_if_basis']=='planned' and r['planned_qty']=={'MNQ':3,'MES':7} and r['warnings']==[]
        assert r['what_if_qty']=={'MNQ':{'BUY':3,'SELL':3},'MES':{'BUY':7,'SELL':7}}
        assert r['margin_total']==pytest.approx(3*6728.37+7*3486.375)
        assert sorted(seen)==sorted([(s,d,q) for s,q in (('MNQ',3),('MES',7)) for d in (1,-1)]
                                    +[(s,d,1) for s in ('MNQ','MES') for d in (1,-1)])
        assert r['margin_day_cap'] is None and 'capped_qty' not in r and r['margin_total_planned']==r['margin_total']
        # $11k per contract: 33k + 77k = 110k > 100k. Since 2026-09-28 night this sizes down (margin day cap)
        # instead of failing: factor 100/110, floor(3 x .909)=2 MNQ, floor(7 x .909)=6 MES -> 22k + 66k = 88k.
        heavy=await preflight(FakePreflight(live,[],[],margin_log([],per=11000.)),manifest=m,wait_seconds=0)
        assert heavy['ok'] and heavy['failures']==[] and heavy['margin_day_cap']=={'MNQ':2,'MES':6}
        assert heavy['margin_total_planned']==pytest.approx(110000.) and heavy['margin_total']==pytest.approx(88000.)
        # $9k per contract: planned 27k + 63k = 90k fits, although the cap (180k) would not: no day cap.
        fits=await preflight(FakePreflight(live,[],[],margin_log([],per=9000.)),manifest=m,wait_seconds=0)
        assert fits['ok'] and fits['margin_total']==pytest.approx(90000.) and fits['margin_day_cap'] is None
        # A planned side above the limit on its own (MES 7 x 15k = 105k) is sized down too, not failed:
        # 45k + 105k = 150k, factor 2/3 -> 2 MNQ / 4 MES = 90k.
        one=await preflight(FakePreflight(live,[],[],margin_log([],per=15000.)),manifest=m,wait_seconds=0)
        assert one['ok'] and one['margin_day_cap']=={'MNQ':2,'MES':4} and one['margin_total']==pytest.approx(90000.)
        # A market the prior-range filter does not arm plans zero and is not previewed at size.
        skip=tuesday_manifest(live,skip_prior_range=True,prior_range_status='OK')
        seen.clear()
        r=await preflight(FakePreflight(live,[],[],margin_log(seen)),manifest=skip,wait_seconds=0)
        assert r['ok'] and r['planned_qty']=={'MNQ':3,'MES':0} and r['margin_total']==pytest.approx(3*6728.37)
        assert ('MES',1,7) not in seen and ('MES',1,1) in seen
        # No quote for a market: planned size unknown, fail closed.
        fp=FakePreflight(live,[],[],margin_log([]));fp.quotes={'NQ':fp.quotes['NQ']}
        r=await preflight(fp,manifest=m,wait_seconds=0)
        assert not r['ok'] and any(f.startswith('PLANNED_SIZE:KeyError') for f in r['failures'])
    asyncio.run(run())

@pytest.mark.parametrize('nq_tr,es_tr',[(565.,79.75),(40.,40.),(2000.,300.),(123.5,17.25),(100.,20.)])
def test_planned_size_equals_size_order(tmp_path,monkeypatch,nq_tr,es_tr):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    from open_breakout.standby import planned_sizes
    live=load_raw(tmp_path,normal_raw())
    m=tuesday_manifest(live,nq_tr,es_tr)
    quotes={'NQ':(24999.75,25000.25,None),'ES':(5999.75,6000.25,None)}
    plan=planned_sizes(live,m,quotes.__getitem__,live.shadow_equity)
    for mk in live.markets:
        mid=sum(quotes[mk.name][:2])/2
        for side,action in [(1,'BUY'),(-1,'SELL')]:
            want=size_order(State(DAY,mk.name,m['markets'][mk.name]['prior_tr'],25),mk,side,mid,mid,live.shadow_equity,live)
            assert plan[mk.execution.symbol][action]==want['qty']<=live.max_contracts_for(mk)
    if (nq_tr,es_tr)==(565.,79.75):
        assert (plan['MNQ']['BUY'],plan['MES']['BUY'])==(3,7)
    if (nq_tr,es_tr)==(100.,20.):
        # Calmest sample ranges: budget-implied sizes stay well under the 60 fat-finger ceiling.
        assert max(plan['MNQ']['BUY'],plan['MES']['BUY'])<60

def test_armed_line_reports_planned_sizes_and_caps(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    from open_breakout.standby import armed_line
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    live=load_raw(tmp_path,normal_raw())
    m=tuesday_manifest(live)
    report=dict(planned_qty={'MNQ':3,'MES':7},margin_total=44589.4,margin_limit=112703.36)
    both={'NQ':NS(phase='FLAT',note=''),'ES':NS(phase='FLAT',note='')}
    line=armed_line(live,report,both,{'NQ':565.,'ES':79.75},76.8,m)
    assert line.startswith("ARMED LIVE: planned 3 MNQ / 7 MES at today's ranges, max 60/60; planned margin $44,589 of limit $112,703;")
    one={'NQ':NS(phase='FLAT',note=''),'ES':NS(phase='SKIPPED',note='PRIOR_RANGE_SKIP: x')}
    line=armed_line(live,report,one,{},76.8,m)
    assert "planned 3 MNQ at today's ranges, max 60;" in line and "NOT ARMED {'ES': 'PRIOR_RANGE_SKIP: x'}" in line
    none={k:NS(phase='SKIPPED',note='x') for k in ('NQ','ES')}
    assert 'NO MARKET' in armed_line(live,report,none,{},76.8,m)

# ---- Margin day cap (owner decision 2026-09-28 night: size down at 09:25 instead of failing) ----

LIVE_LIMIT=112703.36 # 20% of the 2026-09-28 evening ExcessLiquidity

def limited(fake,limit):
    async def values():return {'NetLiquidation':10*limit,'ExcessLiquidity':limit/.2}
    fake.account_values=values;return fake

def test_margin_day_cap_calm_range_sizes_both_markets_down_proportionally(tmp_path,monkeypatch):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    from types import SimpleNamespace as NS
    from open_breakout.standby import preflight, armed_line
    live=load_raw(tmp_path,normal_raw())
    m=tuesday_manifest(live,100.,20.)
    seen=[]
    async def run():
        r=await preflight(limited(FakePreflight(live,[],[],margin_log(seen)),LIVE_LIMIT),manifest=m,wait_seconds=0)
        assert r['ok'],r['failures']
        assert r['margin_limit']==pytest.approx(LIVE_LIMIT) and r['planned_qty']=={'MNQ':20,'MES':23}
        # 20 x 6,728.37 + 23 x 3,486.375 = 214,754.02; factor 112,703.36 / 214,754.02 = 0.5248:
        # floor(20 x .5248) = 10 MNQ, floor(23 x .5248) = 12 MES -> 67,283.70 + 41,836.50 = 109,120.20.
        assert r['margin_total_planned']==pytest.approx(214754.02)
        assert r['margin_day_cap']=={'MNQ':10,'MES':12} and r['capped_qty']=={'MNQ':10,'MES':12}
        assert r['margin_total']==r['margin_total_capped']==pytest.approx(109120.2)
        assert r['margin_day_cap_zero']==[] and r['margin_day_cap_reductions']==0
        # Planned sizes stay reported unchanged; the capped sizes were re-what-iffed, both sides.
        assert r['what_if_qty']=={'MNQ':{'BUY':20,'SELL':20},'MES':{'BUY':23,'SELL':23}}
        assert {(s,d,q) for s,q in (('MNQ',10),('MES',12)) for d in (1,-1)}<=set(seen)
        both={'NQ':NS(phase='FLAT',note=''),'ES':NS(phase='FLAT',note='')}
        line=armed_line(live,r,both,{'NQ':100.,'ES':20.},76.8,m)
        assert line.startswith("ARMED LIVE: planned 20 MNQ / 23 MES, margin-capped to 10 / 12 at today's ranges, max 60/60; "
                               "margin $109,120 of limit $112,703 (uncapped $214,754);")
    asyncio.run(run())

def test_margin_day_cap_zero_when_one_contract_does_not_fit(tmp_path,monkeypatch):
    """NQ TR 2000 plans 1 MNQ at $95k; ES plans 7 MES at $3k. 116k > 100k: MES 6 (18k), and 1 MNQ beside it
    (113k) does not fit, so MNQ is capped to 0 and not armed while MES trades 6."""
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    from open_breakout.standby import preflight, armed_line
    live=load_raw(tmp_path,normal_raw())
    m=tuesday_manifest(live,2000.,79.75)
    per={'MNQ':95000.,'MES':3000.}
    async def run():
        r=await preflight(FakePreflight(live,[],[],lambda mk,b:b['qty']*per[mk.execution.symbol]),manifest=m,wait_seconds=0)
        assert r['ok'],r['failures']
        assert r['planned_qty']=={'MNQ':1,'MES':7} and r['margin_day_cap']=={'MNQ':0,'MES':6}
        assert r['margin_day_cap_zero']==['MNQ'] and r['margin_total']==pytest.approx(18000.)
        assert r['what_if_capped']['MNQ:BUY']['qty']==0
        # The service does not arm MNQ; MES trades at the cap.
        now=[at('09:30:00')]
        b=SimBroker(live,lambda:now[0])
        store=Store(tmp_path/'s.sqlite',live.fingerprint,DAY)
        service=Service(live,m,store,b,lambda:now[0],margin_day_cap={'NQ':0,'ES':6},margin_detail={'margin_limit':1e5})
        try:
            assert service.states['NQ'].phase=='SKIPPED' and service.states['NQ'].note.startswith('MARGIN_DAY_CAP_ZERO')
            for market,base in [('NQ',25000),('ES',6000)]:
                await tick(service,b,now,base,'09:30:00',market)
            for market,base in [('NQ',25000),('ES',6000)]:
                await tick(service,b,now,base+600 if market=='NQ' else base+30,'09:30:01',market)
            entries={o['market'].name:o['body']['qty'] for o in b.orders.values() if o['body']['kind']=='LMT'}
            assert entries=={'ES':6}
            ev=events(store)
            assert [e['body']['market'] for e in ev if e['kind']=='MARGIN_DAY_CAP_ZERO']==['NQ']
            cap=[e['body'] for e in ev if e['kind']=='MARGIN_DAY_CAP']
            assert len(cap)==1 and cap[0]['caps']=={'NQ':0,'ES':6} and cap[0]['margin_limit']==1e5
            assert [(e['body']['market'],e['body']['planned'],e['body']['qty']) for e in ev if e['kind']=='SIZED_DOWN_MARGIN']==[('ES',7,6)]
            line=armed_line(live,r,service.states,{},76.8,m)
            assert "planned 7 MES, margin-capped to 6 at today's ranges, max 60;" in line and 'NOT ARMED' in line and 'MARGIN_DAY_CAP_ZERO' in line
        finally:store.close()
    asyncio.run(run())

def test_margin_day_cap_iterative_reduction_and_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    from open_breakout.standby import preflight, MARGIN_CAP_MAX_REDUCTIONS
    assert MARGIN_CAP_MAX_REDUCTIONS==10
    live=load_raw(tmp_path,normal_raw())
    m=tuesday_manifest(live,100.,20.)
    def surcharge(extra,at_least,at_most=99):
        return lambda mk,b:b['qty']*IB_MARGIN[(mk.execution.symbol,b['side'])]+(extra if at_least<=b['qty']<=at_most else 0.)
    async def run():
        # Nonlinear: +$15k per market at 5+ contracts. Planned 244,754 -> caps 8 / 9 (linear estimate), which
        # re-what-if at 115,204 > 100k; one contract off alternately: MNQ 7 (108,476), MES 8 (104,990), MNQ 6
        # (98,261) fits after three reductions.
        r=await preflight(FakePreflight(live,[],[],surcharge(15000.,5)),manifest=m,wait_seconds=0)
        assert r['ok'],r['failures']
        assert r['margin_day_cap']=={'MNQ':6,'MES':8} and r['margin_day_cap_reductions']==3
        assert r['margin_total']==pytest.approx(6*6728.37+8*3486.375+30000.)
        # $60k per market at any size: even 1 + 1 (133k) never fits; reductions stop and arming fails closed.
        bad=await preflight(FakePreflight(live,[],[],surcharge(60000.,1)),manifest=m,wait_seconds=0)
        assert not bad['ok'] and bad['margin_day_cap_reductions']<=MARGIN_CAP_MAX_REDUCTIONS
        assert [f for f in bad['failures'] if f.startswith('MARGIN_TOTAL:')]==[f'MARGIN_TOTAL:{bad["margin_total"]:.2f}>100000.00']
        # +$88k per market at 3-15 contracts only (planned 20/23 is linear): caps 9 / 10 re-what-if far over,
        # and ten alternate reductions reach 4 / 5, still over: the limit fails closed although 2 / 2 would fit.
        steep=await preflight(FakePreflight(live,[],[],surcharge(88000.,3,15)),manifest=m,wait_seconds=0)
        assert steep['margin_day_cap']=={'MNQ':4,'MES':5}
        assert not steep['ok'] and steep['margin_day_cap_reductions']==MARGIN_CAP_MAX_REDUCTIONS
        assert any(f.startswith('MARGIN_TOTAL:') for f in steep['failures'])
    asyncio.run(run())

def test_margin_day_cap_clamps_entries_both_sides_and_attempts(config,tmp_path):
    async def run():
        now=[at('09:30:00')]
        b=SimBroker(config,lambda:now[0])
        store=Store(tmp_path/'session.sqlite',config.fingerprint,DAY)
        service=Service(config,manifest(config),store,b,lambda:now[0],margin_day_cap={'NQ':2,'ES':None})
        try:
            assert service.margin_day_cap=={'NQ':2}
            await tick(service,b,now,20000,'09:30:00')
            await tick(service,b,now,20010,'09:30:01') # long, planned 6 -> 2
            s=service.states['NQ'];assert s.side==1 and s.qty==2 and s.planned_risk==pytest.approx(2*23.7)
            await tick(service,b,now,19999,'09:30:02');await service.watchdog() # stopped
            assert s.qty==0
            await tick(service,b,now,19990,'09:30:03') # short, second attempt, planned 6 -> 2
            assert s.side==-1 and s.qty==2 and s.attempts==2
            entries=[o['body'] for o in b.orders.values() if o['body']['kind']=='LMT']
            assert [(e['side'],e['qty']) for e in entries]==[(1,2),(-1,2)]
            down=[e['body'] for e in events(store) if e['kind']=='SIZED_DOWN_MARGIN']
            assert [(d['side'],d['attempt'],d['planned'],d['qty']) for d in down]==[(1,1,6,2),(-1,2,6,2)]
            assert [e['body']['caps'] for e in events(store) if e['kind']=='MARGIN_DAY_CAP']==[{'NQ':2}]
            assert store.get('margin_day_cap')=={'NQ':2} and not service.halted
        finally:store.close()
    asyncio.run(run())

def test_no_margin_day_cap_leaves_sizing_and_journal_unchanged(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            assert service.margin_day_cap=={}
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,20010,'09:30:01')
            assert service.states['NQ'].qty==6
            assert not [e for e in events(store) if e['kind'].startswith(('MARGIN_DAY_CAP','SIZED_DOWN_MARGIN'))]
            assert store.get('margin_day_cap') is None
        finally:store.close()
    asyncio.run(run())

def test_live_send_refuses_above_effective_cap(tmp_path,monkeypatch):
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {LIVE_ACCOUNT}')
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        for pilot,cap in [(10,10),(3,3),(None,60),(60,60)]:
            live=load_raw(tmp_path,normal_raw(pilot=pilot),f'cap{pilot}.json')
            adapter=IBKR(live,session=DAY)
            m=live.markets[0];adapter.contracts[m.execution.con_id]=object()
            sent=[]
            adapter.ib.placeOrder=lambda c,o:sent.append(o) or object()
            body=dict(kind='STP',side=-1,stop=20000,tif='GTC',oca='g',account=live.account,ref='x')
            adapter.send(1,m,{**body,'qty':cap})
            assert sent[-1].totalQuantity==cap
            for bad in [cap+1,0,-1,2.5,float('nan'),True]:
                with pytest.raises(PermissionError,match='quantity cap'):adapter.send(2,m,{**body,'qty':bad})
            assert len(sent)==1
    asyncio.run(run())

@pytest.mark.parametrize('sizing',['one_lot','normal','margin_capped','resting'])
def test_live_session_arms_after_preflight_and_trades(live,tmp_path,monkeypatch,capsys,sizing):
    monkeypatch.setenv('EXECUTION_OWNER_REGISTRY',str(tmp_path/'owners'))
    import sqlite3
    from types import SimpleNamespace as NS
    from open_breakout import standby
    from open_breakout.__main__ import main_async
    if sizing in {'normal','margin_capped','resting'}:
        live=load_raw(tmp_path,{**normal_raw(),'entry_order_type':'stop_limit' if sizing=='resting' else 'ioc'})
    clock=[at('08:59:59')]
    class ClockDateTime(datetime):
        @classmethod
        def now(cls,tz=None):return clock[0].astimezone(tz) if tz else clock[0]
    monkeypatch.setattr(standby,'datetime',ClockDateTime)
    monkeypatch.setattr(standby,'webhook_url',lambda:None)
    placed=[]
    class Feed(SimBroker):
        instance=None
        def __init__(self,c,session=None):
            super().__init__(c,lambda:clock[0])
            Feed.instance=self;self.session=session;self.healthy=True;self.connected=False
            self.contract_report={};self.data_types={};self.last_seen={};self.ack=None
            async def empty():return []
            self.ib=NS(client=NS(placeOrder=lambda *a:placed.append(a[0])),isConnected=lambda:self.connected,
                       reqPositionsAsync=empty,reqAllOpenOrdersAsync=empty)
        async def connect(self):
            self.ack=self.config.authorize(self.session)
            with pytest.raises(PermissionError):self.ib.client.placeOrder(0,None,NS(whatIf=False))
            self.connected=True
        def subscribe(self,on_tick,capture):
            self.on_tick=on_tick;self.capture=capture
            self.last_seen={f'{m.name}:{k}':clock[0] for m in live.markets for k in ('trade','quote')}
        async def history(self):
            return {m.name:pd.concat([minutes('2026-09-22'),minutes('2026-09-23')]) for m in live.markets}
        async def account_values(self):return {'NetLiquidation':600000.,'ExcessLiquidity':500000.}
        async def what_if(self,m,body):
            # margin_capped: $3,000 per contract, so the planned 47 + 13 (180k) exceeds the 100k limit.
            margin=3000*body['qty'] if sizing=='margin_capped' else 3000
            return NS(initMarginChange=str(margin),maintMarginChange='2000',commission=0.,warningText='')
        def send(self,oid,market,body):
            self.config.authorize(self.session)
            self.ib.client.placeOrder(oid,None,NS(whatIf=False))
            super().send(oid,market,body)
        def close(self):self.connected=False
    monkeypatch.setattr(standby,'IBKR',Feed)
    original_sleep=asyncio.sleep
    times=iter(['09:00:00','09:25:00','09:30:00','09:30:01','15:55:00','16:01:00'])
    async def advance(seconds):
        stamp=next(times);clock[0]=at(stamp)
        feed=Feed.instance
        feed.last_seen={k:clock[0] for k in feed.last_seen}
        if stamp<'09:30:00':
            # Pre-open quotes stream as they do live; the 09:25 preflight sizes off their mid.
            for m,base in [('NQ',20000),('ES',5000)]:feed.update_quote(m,base-.25,base,clock[0])
        if '09:30:00'<=stamp<'16:01:00':
            for m,base in [('NQ',20000),('ES',5000)]:
                price=base+(10 if stamp>'09:30:00' else 0)
                feed.update_quote(m,price-.25,price,clock[0])
                feed.capture(dict(kind='quote',market=m,time=clock[0].isoformat(),bid=price-.25,ask=price))
                feed.on_tick(m,clock[0],price)
                if sizing=='resting':feed.update_trade(m,price,clock[0])
        for _ in range(20):await original_sleep(0)
    monkeypatch.setattr(standby.asyncio,'sleep',advance)
    risk=tmp_path/'risk.parquet'
    pd.DataFrame({'63d':25.},index=pd.bdate_range('2026-09-01','2026-09-23')).to_parquet(risk)
    run=tmp_path/'run'
    asyncio.run(main_async(NS(command='live-session',config=str(tmp_path/'live.json'),
        session=DAY,risk_parquet=str(risk),state_dir=str(run),roll_verified=True)))
    with sqlite3.connect(run/'runtime.sqlite') as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT * FROM meta')}
    assert meta['phase']=='SESSION_COMPLETE' and meta['mode']=='live'
    assert meta['live_ack']==f'LIVE {DAY} {LIVE_ACCOUNT}'
    assert meta['preflight_connect']['ok'] and meta['preflight_arm']['ok']
    with sqlite3.connect(run/'trades.sqlite') as db:
        orders=[(r[0],json.loads(r[1])) for r in db.execute('SELECT role,body FROM orders')]
        ack=json.loads(db.execute("SELECT value FROM meta WHERE key='live_ack'").fetchone()[0])
    assert ack==meta['live_ack']
    entries=[b for role,b in orders if role=='ENTRY']
    if sizing=='resting':
        assert len(entries)==4 and all(b['kind']=='STP LMT' and b['tif']=='GTD' for b in entries)
        entries=[b for b in entries if b['side']==1]
    assert len(placed)==len(orders) and all(b['account']==LIVE_ACCOUNT for _,b in orders)
    armed=[l for l in capsys.readouterr().out.splitlines() if 'ARMED LIVE' in l]
    if sizing=='one_lot':
        # NQ sizes to 6 micros unclamped -> 1; ES budget (5bp of 100k) is below one MES of risk -> no trade.
        assert len(entries)==1 and entries[0]['qty']==1 and entries[0]['ref'].startswith('MNQ|BUY|OpenBreakout|')
        assert [role for role,_ in orders]==['ENTRY','STOP','TIME']
        # Live writes its own day R too: one timed-exit NQ trade, no ES trade.
        assert meta['day_r']['NQ']['trades']==1 and meta['day_r']['ES']['r'] is None
        assert len(armed)==1 and "planned 1 MNQ / 0 MES at today's ranges, max 1/1" in armed[0]
        assert meta['preflight_connect']['what_if_basis']=='reference' and meta['preflight_arm']['what_if_basis']=='planned'
    elif sizing=='margin_capped':
        # Planned 47 / 13 at $3k each = 180k > 100k: factor 5/9 -> 26 MNQ / 7 MES = 99k; the session arms and
        # both entries are clamped to the day cap (stop and timed exit follow the fill).
        arm=meta['preflight_arm']
        assert arm['ok'] and arm['margin_day_cap']=={'MNQ':26,'MES':7} and arm['capped_qty']=={'MNQ':26,'MES':7}
        assert arm['planned_qty']=={'MNQ':47,'MES':13}
        assert arm['margin_total_planned']==pytest.approx(180000.) and arm['margin_total']==pytest.approx(99000.)
        assert meta['margin_day_cap']=={'NQ':26,'ES':7}
        assert sorted((b['ref'].split('|')[0],b['qty']) for b in entries)==[('MES',7),('MNQ',26)]
        assert sorted(b['qty'] for role,b in orders if role in {'STOP','TIME'})==[7,7,26,26]
        assert len(armed)==1 and ("planned 47 MNQ / 13 MES, margin-capped to 26 / 7 at today's ranges, max 60/60; "
                                  "margin $99,000 of limit $100,000 (uncapped $180,000)") in armed[0]
        with sqlite3.connect(run/'trades.sqlite') as db:
            ev=[(k,json.loads(b)) for k,b in db.execute('SELECT kind,body FROM events ORDER BY seq')]
        assert [b['caps'] for k,b in ev if k=='MARGIN_DAY_CAP']==[{'NQ':26,'ES':7}]
        assert sorted((b['market'],b['planned'],b['qty']) for k,b in ev if k=='SIZED_DOWN_MARGIN')==[('ES',13,7),('NQ',47,26)]
    else:
        # $750k basis, prior TR 40 (stop 10 pts): MNQ 1125/23.70 -> 47 and MES 750/56.70 -> 13, unclamped (no
        # per-market cap since 2026-09-28 late; 60 is the fat-finger ceiling). 1,113.90 + 737.10 fits the 25bp
        # open cap ($1,875), so neither is sized down.
        assert sorted((b['ref'].split('|')[0],b['qty']) for b in entries)==[('MES',13),('MNQ',47)]
        assert sorted(role for role,_ in orders)==['ENTRY']*(4 if sizing=='resting' else 2)+['STOP','STOP','TIME','TIME']
        # Stop and timed exit carry the filled quantity.
        assert sorted(b['qty'] for role,b in orders if role in {'STOP','TIME'})==[13,13,47,47]
        assert meta['day_r']['NQ']['trades']==1 and meta['day_r']['ES']['trades']==1
        assert meta['settings']['effective_max_contracts']=={'MNQ':60,'MES':60}
        assert len(armed)==1 and "planned 47 MNQ / 13 MES at today's ranges, max 60/60" in armed[0]
        assert meta['preflight_arm']['what_if_qty']=={'MNQ':{'BUY':47,'SELL':47},'MES':{'BUY':13,'SELL':13}}
        assert meta['preflight_arm']['margin_total']==pytest.approx(6000.)
        assert meta['preflight_arm']['margin_day_cap'] is None and meta['margin_day_cap'] is None
        assert "planned 47 MNQ / 13 MES at today's ranges, max 60/60; planned margin $6,000 of limit $100,000;" in armed[0]


# ---- Adversarial review fixes ----

def collect(service):
    sent=[];service.notify=sent.append;return sent

def test_halt_alerts_every_new_reason_and_again_with_open_position(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await open_nq(service,b,now)
            sent=collect(service)
            s=service.states['NQ'];qty=s.qty
            s.qty=0
            service.halt('A');service.halt('A');service.halt('B')
            assert sent==['HALTED (shadow): A','HALTED (shadow): B']
            s.qty=qty
            service.halt('A') # same reason, now with a position open: alert again
            assert len(sent)==3 and 'WITH OPEN POSITION' in sent[-1]
            service.halt('A');assert len(sent)==3
            service.halt('C');assert len(sent)==4
        finally:store.close()
    asyncio.run(run())

def test_alert_flush_waits_for_pending_posts(monkeypatch):
    import time as systime
    from open_breakout.alerts import Alerts
    done=[]
    def slow(self,text):systime.sleep(.2);done.append(text)
    monkeypatch.setattr(Alerts,'_post',slow)
    a=Alerts('t','https://example.invalid/hook')
    a('one');a('two');a.flush(5.)
    assert len(done)==2 and not a.threads

def test_slack_is_opt_in_only(monkeypatch):
    """Owner decision 2026-09-28: no Slack for Open Breakout unless OPEN_BREAKOUT_SLACK=1."""
    from open_breakout.alerts import webhook_url
    monkeypatch.setenv('SLACK_WEBHOOK_URL','https://example.invalid/hook')
    monkeypatch.delenv('OPEN_BREAKOUT_SLACK',raising=False)
    assert webhook_url() is None
    monkeypatch.setenv('OPEN_BREAKOUT_SLACK','0')
    assert webhook_url() is None
    monkeypatch.setenv('OPEN_BREAKOUT_SLACK','1')
    assert webhook_url()=='https://example.invalid/hook'

@pytest.mark.parametrize('code',[10349,161,10148,None])
def test_synthesized_cancel_without_reject_code_halts_but_never_flattens(config,tmp_path,code):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        try:
            sent=collect(service)
            await open_nq(service,b,now)
            s=service.states['NQ'];stop=s.stop_order
            if code:b.error_codes[stop]=code
            b.orders[stop]['status']='Cancelled';b.status_callback(stop,'Cancelled')
            await service.drain()
            assert not flatten_orders(b) and s.qty==6 and service.halted
            assert 'PROTECTIVE_STOP_CANCELLED' in store.get('halt_reason') and not store.get('flattened')
            assert any('FLATTEN BY HAND' in m for m in sent)
            assert b.status(s.time_order)=='Submitted' # nothing cancelled on an ambiguous signal
        finally:store.close()
    asyncio.run(run())

class TimedExitFillsOnCancel(SimBroker):
    """The flatten's cancel of the timed exit loses the race: the order fills, cancel confirms later."""
    def cancel(self,oid):
        o=self.orders[oid]
        if o['body']['kind']=='MKT' and o['status']=='Submitted':
            bid,ask,_=self.quotes[o['market'].name]
            self.execute(oid,o['body']['qty'],bid if o['body']['side']==-1 else ask)
            return
        super().cancel(oid)

def test_flatten_waits_for_cancel_and_sibling_fill_after_cancel_is_not_reversed(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,TimedExitFillsOnCancel)
        service.flatten_delay=0;service.cancel_timeout=.1
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            b.order_error_callback(s.stop_order,201,'Order rejected')
            await service.drain()
            assert not flatten_orders(b) and s.qty==0
            assert b.positions[config.markets[0].execution.con_id]==0 # no reversal
            assert 'Exit exceeds' not in str(store.get('halt_reason'))
        finally:store.close()
    asyncio.run(run())

class IgnoresCancel(SimBroker):
    def cancel(self,oid):pass

def test_unconfirmed_cancel_aborts_flatten(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,IgnoresCancel)
        service.flatten_delay=0;service.cancel_timeout=.05
        try:
            sent=collect(service)
            await open_nq(service,b,now)
            b.order_error_callback(service.states['NQ'].stop_order,201,'Order rejected')
            await service.drain()
            assert not flatten_orders(b) and service.states['NQ'].qty==6
            assert 'UNCONFIRMED_CANCEL' in store.get('halt_reason')
            assert any('FLATTEN BY HAND' in m for m in sent)
        finally:store.close()
    asyncio.run(run())

def test_exit_fill_cancels_sibling_explicitly(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            await tick(service,b,now,19999,'09:30:02') # stop fills
            assert s.qty==0 and s.time_order in service.cancel_requested
            assert b.status(s.time_order)=='Cancelled' and not service.halted
        finally:store.close()
    asyncio.run(run())

def test_orphaned_sibling_after_flat_alerts(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,IgnoresCancel)
        service.sibling_timeout=0
        try:
            sent=collect(service)
            await open_nq(service,b,now)
            s=service.states['NQ'];stop=s.stop_order
            # Stop fills but the broker does not cancel the OCA sibling nor accept our cancel.
            b.orders[s.time_order]['body'].pop('oca');b.orders[stop]['body'].pop('oca')
            b.execute(stop,s.qty,19999.75)
            await service.drain()
            assert service.halted and 'ORPHANED_EXIT_ORDER' in store.get('halt_reason')
            assert any('CANCEL BY HAND' in m for m in sent)
        finally:store.close()
    asyncio.run(run())

def test_working_openbreakout_order_after_1556_halts(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            # Journal flat but the broker still shows our timed exit working.
            b.positions[config.markets[0].execution.con_id]=0
            s.qty=0;s.entry_value=0.;s.phase='FLAT'
            now[0]=at('15:56:30')
            b.quotes['NQ']=(20009.75,20010,now[0])
            await service.watchdog()
            assert service.halted and 'after 15:56' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

def test_watchdog_ignores_fill_in_flight_during_snapshot(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            original=b.snapshot
            async def racing():
                snap=await original()
                b.execute(s.stop_order,s.qty,19999.75) # fill arrives while the snapshot is in flight
                return snap
            b.snapshot=racing
            await service.watchdog()
            b.snapshot=original
            assert not service.halted and service.mismatches==0 and s.qty==0
        finally:store.close()
    asyncio.run(run())

class HaltDuringEntry(SimBroker):
    service=None
    async def wait_terminal(self,oid,timeout):
        HaltDuringEntry.service.halt('FEED_BLIP_DURING_ENTRY')
        await super().wait_terminal(oid,timeout)

def test_timed_exit_sent_even_when_halted_during_entry(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,HaltDuringEntry)
        HaltDuringEntry.service=service
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            assert service.halted and s.qty==6 and s.time_order
            assert store.orders()[s.time_order]['body']['good_after'].endswith('15:55:00 America/New_York')
        finally:store.close()
    asyncio.run(run())

def test_late_entry_fill_is_recorded_protected_and_halts(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,NoFill)
        try:
            sent=collect(service)
            await open_nq(service,b,now)
            s=service.states['NQ'];entry=s.entry_order
            assert s.qty==0 and s.phase=='FLAT'
            b.positions[config.markets[0].execution.con_id]=1
            b.fill_callback(entry,'LATE-1',1,20010.)
            await service.drain()
            assert s.qty==1 and s.stop_order and s.time_order and service.halted
            assert b.orders[s.stop_order]['body']['kind']=='STP' and b.orders[s.stop_order]['body']['qty']==1
            assert 'LATE_ENTRY_EXECUTION' in store.get('halt_reason') and any('LATE ENTRY' in m for m in sent)
        finally:store.close()
    asyncio.run(run())

class MarginWarns(SimBroker):
    async def check_margin(self,*args):raise ValueError('what-if warning')

def test_live_skips_per_entry_what_if(live,tmp_path):
    async def run():
        service,b,store,now=setup_service(live,tmp_path,MarginWarns)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            assert not service.halted and s.qty==1 and s.phase=='OPEN'
        finally:store.close()
    asyncio.run(run())

class StopFillsDuringTimeAck(SimBroker):
    async def wait_ack(self,oid,timeout):
        order=self.orders[oid]
        if order['body']['kind']=='MKT':
            stop=[k for k,o in self.orders.items() if o['body']['kind']=='STP'][0]
            self.execute(stop,self.orders[stop]['body']['qty'],self.orders[stop]['body']['stop'])
        await super().wait_ack(oid,timeout)

def test_timed_exit_cancelled_by_oca_after_stop_fill_is_accepted(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path,StopFillsDuringTimeAck)
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            assert s.qty==0 and not service.halted and b.status(s.time_order)=='Cancelled'
            await service.watchdog();assert s.phase=='FLAT'
        finally:store.close()
    asyncio.run(run())

def test_farm_blips_do_not_halt_and_snapshot_is_serialized(config):
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        from types import SimpleNamespace as NS
        adapter=IBKR(config);halts=[]
        adapter.halt_callback=halts.append
        for code in [2103,2105,2104]:adapter._error(-1,code,'farm',None)
        assert adapter.healthy and not halts
        adapter._error(-1,1100,'lost',None);assert not adapter.healthy
        active=[0];peak=[0]
        async def slow(value):
            active[0]+=1;peak[0]=max(peak[0],active[0]);await asyncio.sleep(.01);active[0]-=1;return value
        adapter.ib=NS(reqAllOpenOrdersAsync=lambda:slow([]),reqPositionsAsync=lambda:slow([]),
                      reqExecutionsAsync=lambda f:slow([]))
        await asyncio.gather(adapter.snapshot(),adapter.snapshot(),adapter.own_position(1),adapter.snapshot())
        assert peak[0]==1
    asyncio.run(run())

def test_watchdog_stale_seconds_config(tmp_path,config):
    assert config.watchdog_stale_seconds==30
    raw=json.loads((tmp_path/'config.json').read_text());raw['watchdog_stale_seconds']=45
    c=load_raw(tmp_path,raw,'wd.json')
    assert c.watchdog_stale_seconds==45 and c.fingerprint!=config.fingerprint
    raw['watchdog_stale_seconds']=0
    with pytest.raises(ValueError):load_raw(tmp_path,raw,'wd0.json')

def test_reconcile_client_id_override_must_differ(live,tmp_path):
    from types import SimpleNamespace as NS
    from open_breakout.__main__ import main_async
    a=NS(command='reconcile',config=str(tmp_path/'live.json'),state=str(tmp_path/'x.sqlite'),session=DAY,client_id=live.client_id)
    with pytest.raises(ValueError,match='client-id'):asyncio.run(main_async(a))

def test_relaunch_allowed_only_when_prior_live_journal_has_no_orders(tmp_path):
    from open_breakout.standby import prior_live_orders
    first=tmp_path/f'{DAY}-live';first.mkdir()
    s=Store(first/'trades.sqlite','fp',DAY);s.close()
    assert prior_live_orders(tmp_path/f'{DAY}-live-2',DAY)==[]
    s=Store(first/'trades.sqlite','fp',DAY);s.order(1,'NQ','ENTRY',{'x':1});s.close()
    assert prior_live_orders(tmp_path/f'{DAY}-live-2',DAY)==[f'{DAY}-live']
    assert prior_live_orders(first.resolve(),DAY)==[]


# ---- Prior-range filter (prereg 2026-09-25) ----

from open_breakout.inputs import continuous_sessions, atr20_from_table, prior_range, range_decision, check_range_fields
from open_breakout.config import RangeFilter

FILTER={'enabled':True,'threshold':1.25,'mode':'skip'}

def filtered(tmp_path,filt=FILTER,name='filtered.json'):
    raw=json.loads((tmp_path/'config.json').read_text());raw['prior_range_filter']=filt
    return load_raw(tmp_path,raw,name)

def test_range_filter_config_validation(config,tmp_path,monkeypatch):
    raw=json.loads((tmp_path/'config.json').read_text())
    assert 'prior_range_filter' not in raw and config.prior_range_filter is None and not config.range_filter_on
    assert Config.load(tmp_path/'config.json').fingerprint==config.fingerprint
    c=filtered(tmp_path)
    assert c.prior_range_filter==RangeFilter(True,1.25,'skip') and c.range_filter_on and c.fingerprint!=config.fingerprint
    off=filtered(tmp_path,{'enabled':False,'threshold':1.25,'mode':'skip'},'off.json')
    assert not off.range_filter_on and off.fingerprint not in {c.fingerprint,config.fingerprint}
    assert filtered(tmp_path,{'enabled':True,'threshold':1.25,'mode':'half'},'half.json').prior_range_filter.mode=='half'
    for bad in [{'enabled':True,'threshold':0.99,'mode':'skip'},{'enabled':True,'threshold':3.01,'mode':'skip'},
                {'enabled':True,'threshold':'1.25','mode':'skip'},{'enabled':True,'threshold':True,'mode':'skip'},
                {'enabled':True,'threshold':1.25,'mode':'quarter'},{'enabled':1,'threshold':1.25,'mode':'skip'},
                {'enabled':True,'threshold':1.25},{'enabled':True,'threshold':1.25,'mode':'skip','x':1},True]:
        with pytest.raises(ValueError):filtered(tmp_path,bad,'bad.json')
    assert filtered(tmp_path,{'enabled':True,'threshold':3,'mode':'skip'},'edge.json').prior_range_filter.threshold==3.
    # Live: skip is accepted; half would floor the one-contract pilot to zero and is rejected.
    ok=load_raw(tmp_path,live_raw(prior_range_filter=FILTER))
    assert ok.mode=='live' and ok.range_filter_on and ok.fingerprint!=load_raw(tmp_path,live_raw(),'plain.json').fingerprint
    with pytest.raises(ValueError,match='half'):load_raw(tmp_path,live_raw(prior_range_filter={**FILTER,'mode':'half'}))

def hourly_session(day,high,low,close,volume):
    """23 one-hour bars, 18:00 ET the evening before to 16:00 ET; the first bar sets the high, the second the low."""
    start=pd.Timestamp(day,tz=NY)-pd.Timedelta(hours=6)
    ix=pd.DatetimeIndex([start+pd.Timedelta(hours=k) for k in range(23)]).tz_convert('UTC')
    f=pd.DataFrame({'open':close,'high':close,'low':close,'close':close,'volume':volume/23},index=ix)
    f.iloc[0,f.columns.get_loc('high')]=high;f.iloc[1,f.columns.get_loc('low')]=low
    return f

def roll_history(vols=None,start='2026-07-20',end='2026-09-23'):
    """Sep contract (prices +5) leads until trade date 09-14, then Dec leads; Sep has no bars after 09-18.
    Every non-roll session has TR 32 except 09-22 (40) and 09-23 (40, the previous session for DAY).
    vols: {trade date: (sep volume, dec volume)} overrides."""
    import exchange_calendars as xcals
    days=[d.date().isoformat() for d in xcals.get_calendar('XNYS').sessions_in_range(start,end)]
    sep,dec=[],[]
    for d in days:
        h,l,c=(120.,80.,110.) if d in {'2026-09-22','2026-09-23'} else (126.,94.,110.)
        vs,vd=(vols or {}).get(d,(1000.,10.) if d<'2026-09-14' else (400.,1000.))
        if d<='2026-09-18':sep.append(hourly_session(d,h+5,l+5,c+5,vs))
        dec.append(hourly_session(d,h,l,c,vd))
    return dict(bar_minutes=60,contracts=[dict(expiry='20260918',bars=pd.concat(sep)),dict(expiry='20261218',bars=pd.concat(dec))])

def test_continuous_sessions_roll_rule_and_atr20():
    entry=roll_history()
    table=continuous_sessions(entry['contracts'])
    # Switch at 00:00 UTC two trade dates after the volume flip (09-14): 09-16 spans both, 09-17 changed contract.
    assert table.loc['2026-09-16','instrument_count']==2 and table.loc['2026-09-17','instrument']=='20261218'
    assert table.loc['2026-09-15','instrument']=='20260918' and table.loc['2026-09-15','instrument_count']==1
    assert list(table.index[table.tr.isna() & (table.index>'2026-08-01')].strftime('%Y-%m-%d'))==['2026-09-16','2026-09-17']
    assert table.loc['2026-09-18','tr']==32 and table.loc['2026-09-15','tr']==32
    atr,window,span=atr20_from_table(table,'2026-09-23')
    # Previous session (09-23) excluded; the window is the 20 valid TRs before it, skipping the two roll sessions.
    assert atr==pytest.approx((19*32+40)/20) and window.index[-1]==pd.Timestamp('2026-09-22')
    assert len(window)==20 and pd.Timestamp('2026-09-16') not in window.index and pd.Timestamp('2026-09-23') not in window.index
    info=prior_range(entry,'2026-09-23','2026-09-22',40.,.25)
    assert info['atr20']==pytest.approx(32.4) and info['ratio']==pytest.approx(40/32.4)
    assert info['roll_excluded']==['2026-09-16','2026-09-17'] and info['atr20_window'][1]=='2026-09-22'
    with pytest.raises(ValueError,match='differs from 1-minute'):prior_range(entry,'2026-09-23','2026-09-22',41.,.25)

def test_previous_session_range_does_not_enter_atr20():
    a=continuous_sessions(roll_history()['contracts'])
    entry=roll_history();bars=entry['contracts'][1]['bars']
    last=bars.index[bars.index>=pd.Timestamp('2026-09-22 22:00',tz='UTC')]
    bars.loc[last[0],'high']=500.
    b=continuous_sessions(entry['contracts'])
    assert b.loc['2026-09-23','tr']>a.loc['2026-09-23','tr']
    assert atr20_from_table(a,'2026-09-23')[0]==atr20_from_table(b,'2026-09-23')[0]

def test_atr20_fails_closed_on_19_valid_ambiguous_roll_and_gaps():
    table=continuous_sessions(roll_history()['contracts'])
    valid=table.tr.dropna()
    with pytest.raises(ValueError,match='only 19 valid'):atr20_from_table(table,valid.index[19])
    # With exactly 20, the session supplying the first prev_close is the unknown-front start of history: fail closed.
    with pytest.raises(ValueError,match='ambiguous'):atr20_from_table(table,valid.index[20])
    assert atr20_from_table(table,valid.index[21])[0]==pytest.approx(valid.iloc[1:21].mean())
    # Only a deciding date with no strict volume leader (exact tie) is ambiguous.
    with pytest.raises(ValueError,match='ambiguous roll'):
        atr20_from_table(continuous_sessions(roll_history({'2026-09-14':(1000.,1000.)})['contracts']),'2026-09-23')
    entry=roll_history();bars=entry['contracts'][0]['bars']
    entry['contracts'][0]['bars']=bars.drop(bars.index[(bars.index>=pd.Timestamp('2026-09-09 14:00',tz='UTC'))&(bars.index<pd.Timestamp('2026-09-09 16:00',tz='UTC'))])
    with pytest.raises(ValueError,match='incomplete session 2026-09-09: 2 missing'):atr20_from_table(continuous_sessions(entry['contracts']),'2026-09-23')
    # 23 bars, but off the hour: the hourly rule checks bar starts, not a count.
    entry=roll_history();bars=entry['contracts'][0]['bars']
    day=(bars.index>=pd.Timestamp('2026-09-08 22:00',tz='UTC'))&(bars.index<pd.Timestamp('2026-09-09 21:00',tz='UTC'))
    assert day.sum()==23
    entry['contracts'][0]['bars']=bars.set_axis(bars.index.where(~day,bars.index+pd.Timedelta(minutes=30)))
    with pytest.raises(ValueError,match='incomplete session 2026-09-09'):atr20_from_table(continuous_sessions(entry['contracts']),'2026-09-23')

def test_near_tie_roll_follows_strict_volume_leader():
    # A 1.1% volume lead (the Sep 2025 NQ pattern) decides the roll exactly as a large lead does: no ambiguity.
    base=continuous_sessions(roll_history()['contracts'])
    near=continuous_sessions(roll_history({'2026-09-14':(989.,1000.)})['contracts'])
    assert not near.ambiguous.loc['2026-08-01':].any()
    assert near.tr.equals(base.tr) and near.instrument.equals(base.instrument)
    assert atr20_from_table(near,'2026-09-23')[0]==pytest.approx(atr20_from_table(base,'2026-09-23')[0])
    # One more contract of volume the other way keeps Sep front for that deciding date: the switch moves one date later.
    later=continuous_sessions(roll_history({'2026-09-14':(1000.,999.)})['contracts'])
    assert list(later.index[later.tr.isna()&(later.index>'2026-08-01')].strftime('%Y-%m-%d'))==['2026-09-17','2026-09-18']
    assert atr20_from_table(later,'2026-09-23')[0]>0

def test_volume_lead_moving_back_is_followed_like_the_research():
    # Dec leads on trade date 09-10 only, Sep again on 09-11, Dec from 09-14: the research series switches
    # to Dec and back, and every session touching a switch has no TR. Not a fail-closed condition.
    t=continuous_sessions(roll_history({'2026-09-10':(500.,600.)})['contracts'])
    assert list(t.index[t.tr.isna()&(t.index>'2026-08-01')].strftime('%Y-%m-%d'))==['2026-09-14','2026-09-15','2026-09-16','2026-09-17']
    assert t.loc['2026-09-14','instrument']=='20261218' and t.loc['2026-09-15','instrument_count']==2
    atr,window,span=atr20_from_table(t,'2026-09-23')
    assert len(window)==20 and atr==pytest.approx((19*32+40)/20) and not span.ambiguous.any()

def test_shortened_holiday_session_counts_and_is_not_completeness_checked():
    # Labor Day 2026-09-07: CME trades 18:00 Sunday to 13:00 Monday (19 hourly bars); XNYS is closed.
    entry=roll_history()
    hol=hourly_session('2026-09-07',150.,94.,110.,1000.).iloc[:19]
    entry['contracts'][0]['bars']=pd.concat([entry['contracts'][0]['bars'],hol+[5.,5.,5.,5.,0.]]).sort_index()
    entry['contracts'][1]['bars']=pd.concat([entry['contracts'][1]['bars'],hol.assign(volume=1.)]).sort_index()
    t=continuous_sessions(entry['contracts'])
    assert t.loc['2026-09-07','missing']==4 and t.loc['2026-09-07','tr']==56
    atr,window,span=atr20_from_table(t,'2026-09-23')
    assert pd.Timestamp('2026-09-07') in window.index and atr==pytest.approx((18*32+40+56)/20)

def test_minute_completeness_rule_unchanged_by_shared_helper():
    from open_breakout.inputs import missing_bars
    f=minutes('2026-09-23')
    halt=(f.index.tz_convert(NY).time>=datetime.strptime('16:15','%H:%M').time())&(f.index.tz_convert(NY).time<datetime.strptime('16:30','%H:%M').time())
    assert halt.sum()==15 and session_range(f[~halt],'2026-09-23')==(120,80,110)
    with pytest.raises(ValueError,match='1 missing'):session_range(f.drop(f.index[halt][-1]+pd.Timedelta(minutes=1)),'2026-09-23')
    assert missing_bars(pd.DatetimeIndex([]),'2026-09-23',60)[0]==pd.Timestamp('2026-09-22 18:00',tz=NY)
    assert len(missing_bars(pd.DatetimeIndex([]),'2026-09-23',60))==23

def test_prior_range_rejects_failed_or_empty_history():
    entry=roll_history()
    with pytest.raises(ValueError,match='boom'):prior_range(dict(error='TimeoutError: boom'),'2026-09-23','2026-09-22',40.,.25)
    # The Sep contract traded inside the window: an empty response is a failed request, not zero volume.
    entry['contracts'][0]['bars']=entry['contracts'][0]['bars'].iloc[:0]
    with pytest.raises(ValueError,match='traded inside the window'):prior_range(entry,'2026-09-23','2026-09-22',40.,.25)
    # An expiry that ended before the window legitimately has no bars.
    later=roll_history(start='2026-09-01');later['contracts'][0]['bars']=later['contracts'][0]['bars'].iloc[:0]
    later['contracts'][0]['expiry']='20260618'
    with pytest.raises(ValueError,match='valid prior TRs'):prior_range(later,'2026-09-23','2026-09-22',40.,.25)

def test_range_history_isolates_market_failures(config):
    pytest.importorskip('ib_insync')
    from types import SimpleNamespace as NS
    from ib_insync import BarData
    from open_breakout.ibkr import IBKR
    calls=[]
    def details(contract):
        async def go():
            if contract.symbol=='NQ':raise ConnectionError('details down')
            m=[m for m in config.markets if m.signal.symbol==contract.symbol][0]
            return [NS(contract=NS(lastTradeDateOrContractMonth=e,conId=c,localSymbol=e,includeExpired=False))
                    for e,c in [('20260918',1),('20261218',m.signal.con_id)]]
        return go()
    def hist(c,**kw):
        async def go():
            calls.append(c.lastTradeDateOrContractMonth)
            if c.lastTradeDateOrContractMonth=='20260918' and calls.count('20260918')==1:return []
            return [BarData(date=datetime(2026,9,22,22,tzinfo=NY),open=1.,high=2.,low=1.,close=1.5,volume=10.)]
        return go()
    async def run():
        adapter=IBKR(config)
        adapter.ib=NS(reqContractDetailsAsync=details,reqHistoricalDataAsync=hist)
        return await adapter.range_history()
    out=asyncio.run(run())
    assert 'details down' in out['NQ']['error']
    es=out['ES']['contracts']
    # The empty first response for the earlier expiry was retried once.
    assert calls==['20260918','20260918','20261218'] and [c['expiry'] for c in es]==['20260918','20261218']
    assert all(len(c['bars'])==1 for c in es)

def test_manifest_one_market_range_failure_leaves_other_market(config,tmp_path):
    c=filtered(tmp_path)
    risk,bars,rb=range_manifest_inputs(c)
    rb['NQ']=dict(error='ValueError: NQ: signal contract not in the expiry list')
    b=build_manifest(c,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=rb)
    assert b['markets']['NQ']['prior_range_status']=='UNAVAILABLE' and b['markets']['NQ']['skip_prior_range']
    assert 'expiry list' in b['markets']['NQ']['prior_range_reason']
    assert b['markets']['ES']['prior_range_status']=='OK' and not b['markets']['ES']['skip_prior_range']

@pytest.mark.parametrize('ratio,skip',[(1.2499,False),(1.25,True),(1.9,True)])
def test_range_decision_threshold(ratio,skip):
    info=dict(atr20=10.,ratio=ratio,atr20_window=['a','b'],roll_excluded=[])
    d=range_decision(RangeFilter(True,1.25,'skip'),info)
    assert d['prior_range_status']=='OK' and d['skip_prior_range'] is skip and d['half_prior_range'] is False
    h=range_decision(RangeFilter(True,1.25,'half'),info)
    assert h['half_prior_range'] is skip and h['skip_prior_range'] is False
    # Disabled or absent: recorded, never acted on.
    for f in [None,RangeFilter(False,1.25,'skip')]:
        d=range_decision(f,info);assert d['ratio']==ratio and not d['skip_prior_range'] and not d['half_prior_range']
    u=range_decision(RangeFilter(True,1.25,'half'),'only 19 valid prior TRs (need 20)')
    assert u['prior_range_status']=='UNAVAILABLE' and u['skip_prior_range'] and u['ratio'] is None and '19' in u['prior_range_reason']
    assert not range_decision(None,'boom')['skip_prior_range']

def range_manifest_inputs(config):
    risk=pd.DataFrame({'63d':25.},index=pd.bdate_range('2026-09-01','2026-09-23'))
    bars={m.name:pd.concat([minutes('2026-09-22'),minutes('2026-09-23')]) for m in config.markets}
    return risk,bars,{m.name:roll_history() for m in config.markets}

@pytest.mark.parametrize('threshold,skip',[(1.2,True),(1.25,False)])
def test_manifest_records_ratio_and_skip_flag(config,tmp_path,threshold,skip):
    c=filtered(tmp_path,{**FILTER,'threshold':threshold})
    risk,bars,rb=range_manifest_inputs(c)
    b=build_manifest(c,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=rb)
    for name,item in b['markets'].items():
        assert item['prior_tr']==40 and item['prior_range_status']=='OK' and item['atr20']==pytest.approx(32.4)
        assert item['ratio']==pytest.approx(40/32.4) and item['skip_prior_range'] is skip
    assert b['prior_range_filter']=={'enabled':True,'threshold':threshold,'mode':'skip'}
    p=tmp_path/'inputs.json';p.write_text(json.dumps(b));assert load_manifest(p,c)==b
    # A tampered flag is caught by the hash, and an inconsistent flag by the range check.
    item=dict(b['markets']['NQ']);item['skip_prior_range']=not skip
    with pytest.raises(ValueError):check_range_fields(c,item)

def test_manifest_fails_closed_when_history_missing_or_short(config,tmp_path):
    c=filtered(tmp_path)
    risk,bars,rb=range_manifest_inputs(c)
    b=build_manifest(c,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_error='TimeoutError: x')
    assert all(i['prior_range_status']=='UNAVAILABLE' and i['skip_prior_range'] for i in b['markets'].values())
    assert 'TimeoutError' in b['markets']['NQ']['prior_range_reason']
    short={k:roll_history(start='2026-08-26') for k in rb}
    b=build_manifest(c,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=short)
    assert all(i['skip_prior_range'] and 'valid prior TRs' in i['prior_range_reason'] for i in b['markets'].values())
    p=tmp_path/'inputs.json';p.write_text(json.dumps(b));assert load_manifest(p,c)==b
    bad={k:dict(v) for k,v in b['markets'].items()};bad['NQ']['skip_prior_range']=False
    with pytest.raises(ValueError,match='Unavailable'):check_range_fields(c,bad['NQ'])

def test_shadow_records_ratio_without_skipping(config,tmp_path):
    risk,bars,rb=range_manifest_inputs(config)
    b=build_manifest(config,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=rb)
    assert b['prior_range_filter'] is None
    for item in b['markets'].values():
        assert item['ratio']==pytest.approx(40/32.4) and item['atr20']==pytest.approx(32.4)
        assert item['skip_prior_range'] is False and item['half_prior_range'] is False
    # Even a ratio far above any threshold never skips with the filter absent.
    b=build_manifest(config,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_error='x')
    assert not any(i['skip_prior_range'] for i in b['markets'].values())
    p=tmp_path/'inputs.json';p.write_text(json.dumps(b));assert load_manifest(p,config)==b
    forged=dict(b['markets']['NQ'],skip_prior_range=True)
    with pytest.raises(ValueError,match='disabled'):check_range_fields(config,forged)

def skip_manifest(config,market='NQ',**fields):
    b=manifest(config);b.pop('hash')
    for name in b['markets']:
        b['markets'][name].update(prior_range_status='OK',atr20=40.,ratio=1.,skip_prior_range=False,
                                  half_prior_range=False,prior_range_reason=None)
    b['markets'][market].update(dict(prior_range_status='OK',atr20=28.,ratio=40/28.,skip_prior_range=True,
                                     half_prior_range=False,prior_range_reason='ratio 1.4286 >= threshold 1.25'),**fields)
    return {**b,'hash':digest(b)}

def test_service_never_arms_skipped_market_and_other_market_trades(config,tmp_path):
    c=filtered(tmp_path)
    async def run():
        now=[at('09:29:00')]
        broker=SimBroker(c,lambda:now[0])
        store=Store(tmp_path/'session.sqlite',c.fingerprint,DAY)
        try:
            service=Service(c,skip_manifest(c,'ES'),store,broker,lambda:now[0])
            nq,es=service.states['NQ'],service.states['ES']
            assert es.phase=='SKIPPED' and es.note.startswith('PRIOR_RANGE_SKIP') and not es.long_armed and not es.short_armed
            events=[(k,json.loads(v)) for k,v in store.db.execute('SELECT kind,body FROM events')]
            skips=[v for k,v in events if k=='PRIOR_RANGE_SKIP']
            assert len(skips)==1 and skips[0]['market']=='ES' and skips[0]['threshold']==1.25 and skips[0]['ratio']==pytest.approx(40/28.)
            for market,base in [('ES',5000),('NQ',20000)]:
                await tick(service,broker,now,base,'09:30:00',market)
                await tick(service,broker,now,base+10,'09:30:01',market)
            assert es.opening==0 and es.attempts==0 and es.phase=='SKIPPED'
            assert nq.opening==20000 and nq.attempts==1 and nq.qty==6 and nq.phase=='OPEN'
            assert not any(o['market']=='ES' for o in store.orders().values())
            await tick(service,broker,now,20010,'09:30:05','NQ');await service.watchdog()
            assert es.phase=='SKIPPED' and not service.halted and nq.phase=='OPEN', store.get('halt_reason')
            assert not any(k=='MARKET_BLOCKED' for k,_ in store.db.execute('SELECT kind,body FROM events'))
            # Restart: the skip persists and is journaled once.
            store.close()
            store2=Store(tmp_path/'session2.sqlite',c.fingerprint,DAY)
            store2.db.execute('INSERT INTO states VALUES (?,?)',('ES',json.dumps(es.record())))
            store2.db.execute('INSERT INTO states VALUES (?,?)',('NQ',json.dumps(State(DAY,'NQ',40.,25).record())))
            again=Service(c,skip_manifest(c,'ES'),store2,SimBroker(c,lambda:now[0]),lambda:now[0])
            assert again.states['ES'].phase=='SKIPPED' and again.states['NQ'].phase=='FLAT'
            assert store2.db.execute("SELECT COUNT(*) FROM events WHERE kind='PRIOR_RANGE_SKIP'").fetchone()[0]==0
            store2.close()
        finally:
            if store.db:store.close()
    asyncio.run(run())

def test_fully_skipped_day_runs_to_close_without_halt(config,tmp_path):
    c=filtered(tmp_path)
    async def run():
        now=[at('09:29:00')]
        broker=SimBroker(c,lambda:now[0])
        store=Store(tmp_path/'both.sqlite',c.fingerprint,DAY)
        try:
            m=skip_manifest(c,'NQ');m.pop('hash')
            m['markets']['ES'].update(prior_range_status='UNAVAILABLE',atr20=None,ratio=None,skip_prior_range=True,
                                      prior_range_reason='UNAVAILABLE: only 19 valid prior TRs (need 20)')
            m={**m,'hash':digest(m)}
            service=Service(c,m,store,broker,lambda:now[0])
            assert {s.phase for s in service.states.values()}=={'SKIPPED'}
            for stamp,price in [('09:30:00',20000),('09:30:01',20100),('09:45:00',20200)]:
                for market in ['NQ','ES']:await tick(service,broker,now,price,stamp,market)
            for stamp in ['09:31:00','11:31:00','15:57:00','16:00:30']:
                now[0]=at(stamp);await service.watchdog()
            assert not service.halted and not store.orders(), store.get('halt_reason')
            assert {s.phase for s in service.states.values()}=={'SKIPPED'}
            kinds={k for k, in store.db.execute('SELECT kind FROM events')}
            assert 'MARKET_BLOCKED' not in kinds and 'HALT' not in kinds
        finally:store.close()
    asyncio.run(run())

def test_service_fails_closed_without_range_decision_when_enabled(config,tmp_path):
    c=filtered(tmp_path)
    now=[at('09:29:00')]
    store=Store(tmp_path/'s.sqlite',c.fingerprint,DAY)
    try:
        service=Service(c,manifest(c),store,SimBroker(c,lambda:now[0]),lambda:now[0])
        assert {s.phase for s in service.states.values()}=={'SKIPPED'}
    finally:store.close()
    # The same legacy manifest under a config without the filter arms both markets.
    store=Store(tmp_path/'t.sqlite',config.fingerprint,DAY)
    try:
        service=Service(config,manifest(config),store,SimBroker(config,lambda:now[0]),lambda:now[0])
        assert {s.phase for s in service.states.values()}=={'FLAT'}
    finally:store.close()

def test_half_mode_halves_contracts_floored(config,tmp_path):
    c=filtered(tmp_path,{**FILTER,'mode':'half'})
    s=State(DAY,'NQ',40.1,25)
    full=size_order(s,c.markets[0],1,20000,20000.25,100000,c)['qty']
    s.half_size=True
    assert size_order(s,c.markets[0],1,20000,20000.25,100000,c)['qty']==full//2
    now=[at('09:29:00')]
    store=Store(tmp_path/'h.sqlite',c.fingerprint,DAY)
    try:
        m=skip_manifest(c,skip_prior_range=False,half_prior_range=True)
        service=Service(c,m,store,SimBroker(c,lambda:now[0]),lambda:now[0])
        assert service.states['NQ'].half_size and service.states['NQ'].phase=='FLAT' and not service.states['ES'].half_size
    finally:store.close()

def test_prepare_cli_prints_prior_range_fields(config,tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace as NS
    import open_breakout.ibkr as ibkr_module
    import open_breakout.__main__ as cli
    c=filtered(tmp_path)
    risk,bars,rb=range_manifest_inputs(c)
    rpath=tmp_path/'risk.parquet';risk.to_parquet(rpath)
    class Fake:
        def __init__(self,config,session=None):
            self.ib=NS(client=NS(placeOrder=lambda *a:None),isConnected=lambda:True)
        async def connect(self):pass
        async def history(self):return bars
        async def range_history(self):return rb
        def close(self):pass
    monkeypatch.setattr(ibkr_module,'IBKR',Fake)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return at('09:00:00').astimezone(tz) if tz else at('09:00:00')
    monkeypatch.setattr(cli,'datetime',Clock)
    out=tmp_path/'prep'/'inputs.json'
    asyncio.run(cli.main_async(NS(command='prepare',config=str(tmp_path/'filtered.json'),session=DAY,
        risk_parquet=str(rpath),out=str(out),roll_verified=True,client_id=None)))
    text=capsys.readouterr().out
    printed=json.loads(text[:text.rindex('}')+1])
    assert printed['prior_range_filter']==FILTER
    for name in ['NQ','ES']:
        p=printed['markets'][name]
        assert set(p)>={'prior_tr','prior_range_status','atr20','ratio','skip_prior_range','prior_range_reason'}
        assert p['prior_range_status']=='OK' and p['atr20']==pytest.approx(32.4) and p['skip_prior_range'] is False
    assert load_manifest(out,c)['markets']['NQ']['ratio']==pytest.approx(40/32.4)


# ---- Shared contracts: orderRef ownership (owner decision 2026-09-27) ----

def legend_order(cid,role='TARGET',oid=77):
    return dict(id=oid,client_id=5,con_id=cid,ref=f'MNQ|SELL|Legend_EMA|{DAY}|{role}',remaining=1,kind='LMT',side=-1,stop=0)

def exec_fill(exec_id,ref,side,shares,cid,account,time=None,client_id=0,order_id=0):
    from types import SimpleNamespace as NS
    return NS(execution=NS(execId=exec_id,acctNumber=account,orderRef=ref,side=side,shares=shares,time=time,
                           clientId=client_id,orderId=order_id,price=1.),contract=NS(conId=cid))

def kinds(store):
    return [k for k, in store.db.execute('SELECT kind FROM events')]

def test_ref_strategy_matches_execution_report_parse_ref():
    from daily_execution_report import parse_ref
    from open_breakout.service import ref_strategy, own_ref
    refs=[f'MNQ|BUY|OpenBreakout|{DAY}|NQ-1-STOP',f'MES|BUY|Legend_EMA|{DAY}',f'MES|SELL|Legend_EMA|{DAY}|TARGET',
          'OB:WHATIF','',None,'a|b|OpenBreakout','x|y| OpenBreakout |d','x|y||d']
    for ref in refs:
        assert ref_strategy(ref)==parse_ref(ref)[0]
    assert own_ref(refs[0]) and not own_ref(refs[1]) and not own_ref(refs[2]) and not own_ref('MNQ|BUY|OpenBreakoutX|d')

def test_attribute_executions_mixed_refs():
    from open_breakout.ibkr import attribute_executions
    ob=lambda k,side='BUY':f'MNQ|{side}|OpenBreakout|{DAY}|NQ-{k}-ENTRY'
    fills=[exec_fill('0001.a.01.01',ob(1),'BOT',1.,101,'U1'),
           exec_fill('0001.b.01.01',ob(1,'SELL'),'SLD',1.,101,'U1'),
           exec_fill('0001.c.01.01',ob(2),'BOT',1.,101,'U1'),
           exec_fill('0001.c.01.01',ob(2),'BOT',1.,101,'U1'), # the same report twice
           exec_fill('0001.d.01.01',f'MES|BUY|Legend_EMA|{DAY}','BOT',1.,103,'U1'),
           exec_fill('0001.e.01.01',f'MNQ|SELL|Legend_EMA|{DAY}|TARGET','SLD',2.,101,'U1'),
           exec_fill('0001.f.01.01','','BOT',5.,101,'U1'), # manual TWS order: neither book
           exec_fill('0001.g.01.01','OB:WHATIF','BOT',5.,101,'U1'),
           exec_fill('0001.h.01.01',ob(3),'BOT',1.,101,'U2'), # another account
           exec_fill('0001.i.01.01',ob(3),'BOT',2.,101,'U1'),
           exec_fill('0001.i.01.02',ob(3),'BOT',1.,101,'U1')] # IB correction replaces .01
    own,other,ids=attribute_executions(fills,'U1')
    assert own=={101:2.} and other=={103:1.,101:-2.}
    assert ids=={'0001.a.01','0001.b.01','0001.c.01','0001.i.01'}
    # Earlier sessions (TWS Trade Log may return seven days) are excluded; today's count.
    from open_breakout.ibkr import session_start
    today=session_start(at('10:00:00'))
    old=[exec_fill('0002.a.01.01',ob(1),'BOT',1.,101,'U1',time=today-timedelta(hours=10)),
         exec_fill('0002.b.01.01',ob(2),'BOT',1.,101,'U1',time=today+timedelta(hours=9.6))]
    own,_,ids=attribute_executions(old,'U1',today)
    assert own=={101:1.} and ids=={'0002.b.01'}

def test_ibkr_own_position_from_executions_cached_and_refreshed_on_own_fill(config):
    from types import SimpleNamespace as NS
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        adapter=IBKR(config);cid=config.markets[0].execution.con_id
        calls=[];fills=[]
        async def execs(f):calls.append(f);return list(fills)
        adapter.ib=NS(reqExecutionsAsync=execs)
        fills.append(exec_fill('1.1.01.01',f'MNQ|BUY|OpenBreakout|{DAY}|NQ-1-ENTRY','BOT',1.,cid,config.account))
        fills.append(exec_fill('1.2.01.01',f'MNQ|BUY|Legend_EMA|{DAY}','BOT',1.,cid,config.account))
        assert await adapter.own_position(cid)==1 and len(calls)==1
        # Account filter only: executions of an earlier process or another client id still count.
        assert calls[0].acctCode==config.account and calls[0].clientId==0
        fills.append(exec_fill('1.3.01.01',f'MNQ|SELL|OpenBreakout|{DAY}|NQ-1-STOP','SLD',1.,cid,config.account))
        assert await adapter.own_position(cid)==1 and len(calls)==1 # cached for one second
        adapter._fill(None,NS(execution=NS(acctNumber=config.account,clientId=config.client_id,orderId=1,
                                           execId='1.3.01.01',shares=1.,price=1.)))
        assert await adapter.own_position(cid)==0 and len(calls)==2 # our own fill invalidates the cache
    asyncio.run(run())

def test_ibkr_executions_race_redelivery_and_failure(config):
    from types import SimpleNamespace as NS
    async def run():
        pytest.importorskip('ib_insync')
        from open_breakout.ibkr import IBKR
        adapter=IBKR(config);cid=config.markets[0].execution.con_id
        ref=f'MNQ|BUY|OpenBreakout|{DAY}|NQ-1-ENTRY'
        mine=exec_fill('2.1.01.01',ref,'BOT',1.,cid,config.account,client_id=config.client_id,order_id=11)
        got=[];adapter.fill_callback=lambda *a:got.append(a)
        gate=asyncio.Event();answer=[[]]
        async def execs(f):
            await gate.wait();return list(answer[0])
        async def empty():return []
        adapter.ib=NS(reqExecutionsAsync=execs,reqAllOpenOrdersAsync=empty,reqPositionsAsync=empty)
        # An own fill lands while a request is in flight: that stale answer is used once, never cached.
        pending=asyncio.ensure_future(adapter.own_position(cid));await asyncio.sleep(0)
        adapter._fill(None,mine);gate.set()
        assert await pending==0 and adapter._exec_books is None
        # ib_insync suppresses the live execDetailsEvent for an execId a reqExecutions answer delivered first:
        # an own execution never handed to the journal is re-delivered once (Service.fill dedups by execId).
        adapter._delivered.clear();got.clear();answer[0]=[mine]
        assert await adapter.own_position(cid)==1 and got==[(11,'2.1.01.01',1.,1.)]
        adapter._exec_books=None
        assert await adapter.own_position(cid)==1 and len(got)==1
        # A failed or slow executions request is UNKNOWN in the snapshot, never an exception.
        async def boom(f):raise TimeoutError('slow')
        adapter.ib.reqExecutionsAsync=boom;adapter._exec_books=None
        snap=await adapter.snapshot()
        assert snap['own'] is None and snap['foreign'] is None and 'slow' in snap['executions_error']
        with pytest.raises(TimeoutError):await adapter.own_position(cid)
    asyncio.run(run())

@pytest.mark.parametrize('foreign',[1,-1])
@pytest.mark.parametrize('held',[False,True])
def test_watchdog_ignores_other_strategy_mid_session(config,tmp_path,capsys,foreign,held):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            await tick(service,b,now,20000,'09:30:00')
            if held:await tick(service,b,now,20010,'09:30:01')
            sent=collect(service)
            await service.watchdog()
            capsys.readouterr()
            # Legend EMA enters: attributed position plus its working TARGET and TIME legs.
            b.foreign_positions[cid]=foreign
            b.foreign_orders=[legend_order(cid),legend_order(cid,'TIME',78)]
            for _ in range(4):await service.watchdog()
            s=service.states['NQ']
            assert not service.halted and service.mismatches==0 and s.qty==(6 if held else 0)
            assert sent==[] and 'RECONCILE_DISCREPANCY' not in kinds(store) and 'HALT' not in kinds(store)
            assert store.get('other_book')['MNQ']['position']==foreign
            b.foreign_positions[cid]=0;b.foreign_orders=[]
            await service.watchdog();await service.watchdog()
            lines=[x for x in capsys.readouterr().out.splitlines() if 'Other book' in x]
            assert len(lines)==2 and 'Legend_EMA' in lines[0] and lines[1].endswith('none')
            assert not service.halted and store.get('other_book')=={}
        finally:store.close()
    asyncio.run(run())

def test_watchdog_halts_on_own_execution_mismatch(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            # OpenBreakout-tagged executions say long 1; the journal says flat.
            b.record(cid,1,f'MNQ|BUY|OpenBreakout|{DAY}|NQ-9-ENTRY')
            await service.watchdog();await service.watchdog();assert not service.halted
            await service.watchdog()
            assert service.halted and 'executions/journal position mismatch' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

@pytest.mark.parametrize('ceiling_halts',[False,True])
def test_account_ceiling_warns_by_default_and_halts_only_when_configured(live,tmp_path,ceiling_halts):
    async def run():
        c=replace(live,ceiling_halts=True) if ceiling_halts else live
        service,b,store,now=setup_service(c,tmp_path)
        cid=live.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            assert service.states['NQ'].qty==1
            sent=collect(service)
            # Legend short 1 nets the account to 0, but its executions are attributed: nothing at all.
            b.foreign_positions[cid]=-1
            for _ in range(4):await service.watchdog()
            assert not service.halted and 'POSITION_CEILING_WARNING' not in kinds(store)
            # Our long 1 flattened in TWS by hand (no orderRef): own executions 1, account 0.
            b.foreign_positions[cid]=0;b.positions[cid]=0
            await service.watchdog();await service.watchdog()
            assert not service.halted and 'POSITION_CEILING_WARNING' not in kinds(store)
            await service.watchdog()
            if ceiling_halts:
                assert service.halted and 'Account position below' in store.get('halt_reason')
            else:
                assert not service.halted and kinds(store).count('POSITION_CEILING_WARNING')==1
                assert any('POSITION CEILING WARNING' in x for x in sent)
                for _ in range(3):await service.watchdog()
                assert not service.halted and kinds(store).count('POSITION_CEILING_WARNING')==1 # once per episode
                assert not flatten_orders(b) and not [o for o in b.orders.values() if o['status']=='Cancelled']
        finally:store.close()
    asyncio.run(run())

@pytest.mark.parametrize('ceiling_halts',[False,True])
def test_invisible_legend_long_against_our_short_never_halts_by_default(live,tmp_path,ceiling_halts):
    """reqExecutions may not return another client's executions: Legend's 09:31 long 1 then shows only in the
    account, which nets our short 1 to 0. Default: a warning, no halt, cancel or flatten."""
    async def run():
        c=replace(live,ceiling_halts=True) if ceiling_halts else live
        service,b,store,now=setup_service(c,tmp_path)
        cid=live.markets[0].execution.con_id
        try:
            await service.watchdog() # 09:30 flat: foreign_base 0
            await tick(service,b,now,20000,'09:30:00');await tick(service,b,now,19990,'09:30:01')
            s=service.states['NQ'];assert s.qty==1 and s.side==-1
            b.foreign_visible=False;b.foreign_positions[cid]=1;b.foreign_orders=[legend_order(cid)]
            for _ in range(4):await service.watchdog()
            if ceiling_halts:
                assert service.halted and 'Account position below' in store.get('halt_reason')
            else:
                assert not service.halted and 'POSITION_CEILING_WARNING' in kinds(store)
                assert b.orders[s.stop_order]['status']=='Submitted' and not flatten_orders(b)
        finally:store.close()
    asyncio.run(run())

def test_own_position_unknown_skips_checks_and_alerts_after_threshold(config,tmp_path):
    async def run():
        # No ticks while the clock advances: widen the stream-gap tolerance so only UNKNOWN is under test.
        service,b,store,now=setup_service(replace(config,watchdog_stale_seconds=300.),tmp_path)
        try:
            await open_nq(service,b,now)
            sent=collect(service)
            # reqExecutions answers without our journaled fills (reconnect, IB limit): own reads 0 vs journal 6.
            saved=list(b.executions);b.executions.clear()
            start=now[0]
            for i in range(29):
                now[0]=start+timedelta(seconds=i);await service.watchdog()
            assert not service.halted and service.mismatches==0 and 'OWN_POSITION_UNKNOWN' not in kinds(store)
            now[0]=start+timedelta(seconds=30);await service.watchdog()
            assert not service.halted and kinds(store).count('OWN_POSITION_UNKNOWN')==1
            assert any('OWN POSITION UNKNOWN' in x and 'NOT halted' in x for x in sent)
            now[0]=start+timedelta(seconds=40);await service.watchdog()
            assert kinds(store).count('OWN_POSITION_UNKNOWN')==1
            # Answer carries our fills again: normal reconciliation resumes.
            b.executions[:]=saved
            await service.watchdog()
            assert 'OWN_POSITION_KNOWN' in kinds(store) and service.own_unknown==0
            assert not service.halted and store.get('heartbeat')['own_known']
        finally:store.close()
    asyncio.run(run())

def test_own_position_request_failure_is_unknown_not_halt(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            await open_nq(service,b,now)
            b.executions_error='TimeoutError: reqExecutions'
            for _ in range(5):await service.watchdog()
            assert not service.halted and service.own_unknown==5 and not store.get('heartbeat')['own_known']
            b.executions_error=None
            await service.watchdog();assert not service.halted and service.own_unknown==0
            # A genuine own mismatch with a trusted answer still halts after three stable checks.
            b.record(config.markets[0].execution.con_id,1,f'MNQ|BUY|OpenBreakout|{DAY}|NQ-9-ENTRY','X-1')
            for _ in range(3):await service.watchdog()
            assert service.halted and 'executions/journal position mismatch' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())

def test_unknown_at_start_does_not_poison_foreign_base(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            b.executions_error='TimeoutError'
            b.positions[cid]=3
            await service.watchdog()
            assert str(cid) not in store.get('foreign_base',{})
            b.executions_error=None;b.positions[cid]=0
            await service.watchdog()
            assert store.get('foreign_base')[str(cid)]==0
        finally:store.close()
    asyncio.run(run())

def test_carried_other_position_is_baselined_not_halted(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            b.positions[cid]=-2 # a hedge carried from before the session; no orderRef in today's executions
            await service.watchdog()
            assert store.get('foreign_base')[str(cid)]==-2
            await open_nq(service,b,now)
            for _ in range(4):await service.watchdog()
            assert service.states['NQ'].qty==6 and not service.halted
        finally:store.close()
    asyncio.run(run())

@pytest.mark.parametrize('foreign',[3,-1])
def test_emergency_flatten_sizes_from_journal_with_foreign_position(config,tmp_path,foreign):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        cid=config.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            b.foreign_positions[cid]=foreign;b.foreign_orders=[legend_order(cid)]
            b.order_error_callback(s.stop_order,201,'Order rejected')
            await service.drain()
            flat=flatten_orders(b)
            assert len(flat)==1 and flat[0][1]['body']['qty']==6 and flat[0][1]['body']['side']==-1
            assert s.qty==0 and b.positions[cid]==0 and b.foreign_positions[cid]==foreign
            assert 'FLATTEN_SKIPPED' not in kinds(store)
        finally:store.close()
    asyncio.run(run())

def test_1556_check_ignores_foreign_orders_and_position(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        cid=config.markets[0].execution.con_id
        try:
            await open_nq(service,b,now)
            await tick(service,b,now,19999,'09:30:02') # stop fills; journal flat
            await service.watchdog();assert service.states['NQ'].phase=='FLAT'
            b.foreign_positions[cid]=1;b.foreign_orders=[legend_order(cid,'TIME')]
            now[0]=at('15:56:30')
            for _ in range(4):await service.watchdog()
            assert not service.halted, store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())


def test_exec_key_is_idempotent_on_ib_execution_ids():
    from open_breakout.service import exec_key
    full='00010198.6ab9ed2b.01.01'
    assert exec_key(full)=='00010198.6ab9ed2b.01'
    assert exec_key(exec_key(full))=='00010198.6ab9ed2b.01'
    assert exec_key('00010198.6ab9ed2b.01.02')==exec_key(full)
    assert exec_key('SIM-7')=='SIM-7'

def test_own_view_trusts_stem_ids_from_the_executions_answer(config,tmp_path):
    """2026-09-28 live regression: the journal stores IB's full execId, the executions answer carries stems
    (attribute_executions keys); stripping twice made every own fill look missing -> OWN_POSITION_UNKNOWN."""
    from open_breakout.service import exec_key
    class IbIds(SimBroker):
        def execute(self,oid,qty,price):
            order=self.orders[oid];body=order['body'];cid=order['market'].execution.con_id
            self.execution+=1
            exec_id=f'00010198.{self.execution:08x}.01.01'
            self.record(cid,body['side']*qty,body['ref'],exec_id)
            self.fill_callback(oid,exec_id,qty,price)
            order['status']='Filled';self.status_callback(oid,'Filled')
        def _own_ids(self):
            return sorted({exec_key(x) for _,_,ref,x in self.executions if x and own_ref(ref)})
    from open_breakout.service import own_ref
    async def run():
        service,b,store,now=setup_service(replace(config,watchdog_stale_seconds=300.),tmp_path,IbIds)
        try:
            await open_nq(service,b,now)
            assert store.fill_ids() and all(i.count('.')==3 for i in store.fill_ids())
            for i in range(3):
                now[0]=now[0]+timedelta(seconds=1);await service.watchdog()
            assert 'OWN_POSITION_UNKNOWN_START' not in kinds(store) and service.own_unknown==0
            assert store.get('heartbeat')['own_known'] and not service.halted
        finally:store.close()
    asyncio.run(run())


# ---- Amendment 2026-09-28: range skip only after a big win ----

from open_breakout.store import journal_day_r, prior_day_r, prior_session_dir

BIG=dict(FILTER,require_prior_big_win=True,big_win_r=2.0)
PREV='2026-09-23'

def trade(market,attempt,side,entries,stop,exits,modified_stop=None):
    """entries [(qty,price)], exits [(role,qty,price)]; modified_stop: a later MODIFY_INTENT stop."""
    return dict(market=market,attempt=attempt,side=side,entries=entries,stop=stop,exits=exits,modified_stop=modified_stop)

def write_journal(path,trades,day=PREV,halted=False):
    s=Store(path,'fp',day)
    try:
        oid=0;fid=0
        def order(t,role,body):
            nonlocal oid
            oid+=1
            s.order(oid,t['market'],role,{**body,'ref':f'MNQ|BUY|OpenBreakout|{day}|{t["market"]}-{t["attempt"]}-{role}'})
            return oid
        def fill(order_id,qty,price):
            nonlocal fid
            fid+=1;s.add_fill(f'X-{fid}',dict(order_id=order_id,qty=qty,price=price))
        for t in trades:
            entry=order(t,'ENTRY',dict(kind='LMT',side=t['side']))
            stop=order(t,'STOP',dict(kind='STP',side=-t['side'],stop=t['stop']))
            if t['modified_stop'] is not None:
                s.event('MODIFY_INTENT',dict(id=stop,body=dict(kind='STP',side=-t['side'],stop=t['modified_stop'])))
            for qty,price in t['entries']:fill(entry,qty,price)
            for role,qty,price in t['exits']:
                fill(stop if role=='STOP' else order(t,role,dict(kind='MKT',side=-t['side'])),qty,price)
        if halted:s.set('halted',True);s.set('halt_reason','X')
    finally:s.close()

def write_runtime(path,day_r,day=PREV):
    s=Store(path,'fp',day)
    try:
        if day_r is not None:s.set('day_r',day_r)
    finally:s.close()

def test_big_win_config_defaults_validation_and_fingerprint(config,tmp_path):
    plain=filtered(tmp_path)
    assert plain.prior_range_filter.require_prior_big_win is False and plain.prior_range_filter.big_win_r==2.
    # Keys absent: same RangeFilter and fingerprint as before the amendment.
    assert plain.prior_range_filter==RangeFilter(True,1.25,'skip')
    assert filtered(tmp_path,dict(FILTER),'again.json').fingerprint==plain.fingerprint
    c=filtered(tmp_path,BIG,'big.json')
    assert c.prior_range_filter==RangeFilter(True,1.25,'skip',True,2.) and c.fingerprint!=plain.fingerprint
    assert filtered(tmp_path,{**FILTER,'big_win_r':0.5},'lo.json').prior_range_filter.big_win_r==.5
    assert filtered(tmp_path,{**FILTER,'require_prior_big_win':True,'big_win_r':5},'hi.json').prior_range_filter.big_win_r==5.
    for bad in [{**BIG,'require_prior_big_win':'yes'},{**BIG,'require_prior_big_win':1},{**BIG,'big_win_r':.49},
                {**BIG,'big_win_r':5.01},{**BIG,'big_win_r':True},{**BIG,'big_win_r':'2'},{**BIG,'big_win_r':float('nan')},
                {**BIG,'extra':1},{'require_prior_big_win':True,'big_win_r':2.}]:
        with pytest.raises(ValueError):filtered(tmp_path,bad,'bad.json')
    live_big=load_raw(tmp_path,live_raw(prior_range_filter=BIG),'livebig.json')
    assert live_big.prior_range_filter.require_prior_big_win and live_big.fingerprint!=load_raw(tmp_path,live_raw(prior_range_filter=FILTER),'l.json').fingerprint

def test_journal_day_r_win_loss_timed_partial_and_attempts(tmp_path):
    write_journal(tmp_path/'t.sqlite',[
        trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,125.)]),              # timed-exit win +2.5R
        trade('NQ',2,-1,[(1,120.)],130.,[('STOP',1,131.)]),            # stopped short, slipped: -1.1R
        trade('ES',1,-1,[(1,50.)],52.,[('STOP',1,52.)]),               # stopped short: -1R
        trade('ES',2,1,[(1,100.),(1,102.)],90.,[('TIME',2,111.)],91.), # partials, stop moved to 91: +1R
    ])
    import sqlite3
    with sqlite3.connect(tmp_path/'t.sqlite') as db:
        out=journal_day_r(db,['NQ','ES','RTY'])
    assert out['NQ']['r']==pytest.approx(1.4) and out['NQ']['trades']==2 and out['NQ']['trades_r']==pytest.approx([2.5,-1.1])
    assert out['ES']['r']==pytest.approx(0.) and out['ES']['trades_r']==pytest.approx([-1.,1.])
    assert out['RTY']['r'] is None and out['RTY']['note']=='no closed trades'

def test_prior_day_r_own_journal_fallback_and_no_trigger_cases(tmp_path):
    # No shadow journal anywhere here: live falls back to its own live journal.
    names=['NQ','ES']
    # Both missing: null, no skip.
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert all(v['r'] is None and v['source'] is None and 'no shadow journal' in v['note'] and 'no live journal' in v['note']
               for v in out.values())
    # Win on NQ, no ES trades; recomputed from fills (no runtime day_r).
    d=tmp_path/f'{PREV}-live';write_journal(d/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,130.)])])
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert out['NQ']['r']==pytest.approx(3.) and out['NQ']['source']=='own_journal' and 'no shadow journal' in out['NQ']['note']
    assert out['ES']['r'] is None and out['ES']['source']=='own_journal' and 'no closed trades' in out['ES']['note']
    # A runtime day_r summary is preferred over recomputation.
    write_runtime(d/'runtime.sqlite',{'NQ':dict(r=-0.5,note='summary'),'ES':dict(r=None,note='none')})
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert out['NQ']['r']==-0.5 and out['NQ']['source']=='own_runtime_meta' and out['NQ']['note'].startswith(f'{PREV}-live: summary')
    assert out['ES']['r'] is None
    # The highest attempt with a journal wins; an attempt without one is ignored.
    d2=tmp_path/f'{PREV}-live-2';write_journal(d2/'trades.sqlite',[trade('ES',1,-1,[(1,50.)],52.,[('STOP',1,52.)])])
    (tmp_path/f'{PREV}-live-3').mkdir();(tmp_path/f'{PREV}-livex').mkdir()
    assert prior_session_dir(tmp_path,PREV,'live')==d2
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert out['ES']['r']==pytest.approx(-1.) and out['NQ']['r'] is None and out['ES']['source']=='own_journal'
    assert out['ES']['note'].startswith(f'{PREV}-live-2:')
    # The shadow session never reads the live journal.
    assert all(v['r'] is None and v['source'] is None for v in prior_day_r(tmp_path,PREV,'shadow',names).values())
    # Halted with no fills, and halted with a position the journal never closed: no trigger.
    h=tmp_path/'h';write_journal(h/f'{PREV}-live'/'trades.sqlite',[],halted=True)
    assert all(v['r'] is None and 'no fills' in v['note'] for v in prior_day_r(h,PREV,'live',names).values())
    o=tmp_path/'o';write_journal(o/f'{PREV}-live'/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[])],halted=True)
    nq=prior_day_r(o,PREV,'live',names)['NQ'];assert nq['r'] is None and 'not closed' in nq['note']
    # A journal for another day, an invalid runtime summary and an unreadable journal fail open to null.
    w=tmp_path/'w';write_journal(w/f'{PREV}-live'/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,130.)])],day='2026-09-22')
    assert 'not 2026-09-23' in prior_day_r(w,PREV,'live',names)['NQ']['note']
    v=tmp_path/'v';write_journal(v/f'{PREV}-live'/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,130.)])])
    write_runtime(v/f'{PREV}-live'/'runtime.sqlite',{'NQ':dict(r='big'),'ES':dict(r=None)})
    assert prior_day_r(v,PREV,'live',names)['NQ']['source']=='own_journal'
    u=tmp_path/'u'/f'{PREV}-live';u.mkdir(parents=True);(u/'trades.sqlite').write_bytes(b'not a database')
    bad=prior_day_r(tmp_path/'u',PREV,'live',names)
    assert all(x['r'] is None and x['source'] is None and 'unreadable' in x['note'] for x in bad.values())

def test_prior_day_r_prefers_the_unfiltered_shadow_journal(tmp_path):
    names=['NQ','ES']
    # Live skipped NQ yesterday (no trades) while the unfiltered shadow won big on it: live reads the shadow.
    live=tmp_path/f'{PREV}-live';write_journal(live/'trades.sqlite',[trade('ES',1,-1,[(1,50.)],52.,[('STOP',1,52.)])])
    sh=tmp_path/f'{PREV}-shadow'
    write_journal(sh/'trades.sqlite',[trade('NQ',1,1,[(6,100.)],90.,[('TIME',6,125.)]),trade('ES',1,-1,[(8,50.)],52.,[('STOP',8,52.5)])])
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert out['NQ']['r']==pytest.approx(2.5) and out['NQ']['source']=='shadow_journal' and out['ES']['r']==pytest.approx(-1.25)
    assert out['NQ']['note'].startswith(f'{PREV}-shadow:')
    # The shadow session reads the same journal with the same label.
    assert prior_day_r(tmp_path,PREV,'shadow',names)==out
    # The shadow's runtime summary is preferred when present.
    write_runtime(sh/'runtime.sqlite',{'NQ':dict(r=2.4,note='summary'),'ES':dict(r=-1.2,note='summary')})
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert out['NQ']['r']==2.4 and out['NQ']['source']=='shadow_runtime_meta'
    # Highest shadow attempt with a journal; one with no fills (halted before trading) falls back to live.
    sh2=tmp_path/f'{PREV}-shadow-2';write_journal(sh2/'trades.sqlite',[],halted=True)
    assert prior_session_dir(tmp_path,PREV,'shadow')==sh2
    out=prior_day_r(tmp_path,PREV,'live',names)
    assert out['ES']['r']==pytest.approx(-1.) and out['ES']['source']=='own_journal' and out['NQ']['r'] is None
    assert f'{PREV}-shadow-2 journal has no fills' in out['NQ']['note']
    shadow_only=prior_day_r(tmp_path,PREV,'shadow',names)
    assert all(v['r'] is None and v['source'] is None for v in shadow_only.values())
    # A shadow that halted after trading: its fills are used; an unclosed position is excluded, not a fallback.
    h=tmp_path/'h';write_journal(h/f'{PREV}-live'/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,99.)])])
    write_journal(h/f'{PREV}-shadow'/'trades.sqlite',[trade('NQ',1,1,[(6,100.)],90.,[('STOP',6,120.)]),
                                                      trade('ES',1,1,[(8,50.)],48.,[])],halted=True)
    out=prior_day_r(h,PREV,'live',names)
    assert out['NQ']['r']==pytest.approx(2.) and out['NQ']['source']=='shadow_journal'
    assert out['ES']['r'] is None and out['ES']['source']=='shadow_journal' and 'not closed' in out['ES']['note']
    # An unreadable shadow journal falls back to live; the reason is kept in the note.
    u=tmp_path/'u';write_journal(u/f'{PREV}-live'/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,130.)])])
    (u/f'{PREV}-shadow').mkdir();(u/f'{PREV}-shadow'/'trades.sqlite').write_bytes(b'not a database')
    out=prior_day_r(u,PREV,'live',names)
    assert out['NQ']['r']==pytest.approx(3.) and out['NQ']['source']=='own_journal' and 'unreadable' in out['NQ']['note']

def test_big_win_on_shadow_skips_live_after_a_live_skip(config,tmp_path):
    # End to end through build_manifest: the live NQ was skipped yesterday (no trades) but the shadow won +3R.
    from types import SimpleNamespace as NS
    from open_breakout.standby import prior_day_inputs
    c=filtered(tmp_path,{**BIG,'threshold':1.2},'big.json')
    previous,_=calendar_dates(DAY)
    runs=tmp_path/'runs'
    write_journal(runs/f'{previous}-live'/'trades.sqlite',[],day=previous)
    write_journal(runs/f'{previous}-shadow'/'trades.sqlite',[trade('NQ',1,1,[(4,100.)],90.,[('TIME',4,130.)])],day=previous)
    prior=prior_day_inputs(NS(mode='live',markets=c.markets),DAY,runs)
    assert prior['NQ']['source']=='shadow_journal' and prior['NQ']['r']==pytest.approx(3.)
    risk,bars,rb=range_manifest_inputs(c)
    b=build_manifest(c,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=rb,prior_day=prior)
    nq=b['markets']['NQ']
    assert nq['skip_prior_range'] is True and nq['prior_big_win'] is True and nq['prior_day_source']=='shadow_journal'
    assert '(shadow_journal) >= big win' in nq['prior_range_reason']
    p=tmp_path/'inputs.json';p.write_text(json.dumps(b));assert load_manifest(p,c)==b
    for bad in ['/some/path/trades.sqlite','runtime_meta']:
        with pytest.raises(ValueError,match='prior_day_source'):check_range_fields(c,dict(nq,prior_day_source=bad))
    with pytest.raises(ValueError,match='without a prior_day_source'):check_range_fields(c,dict(nq,prior_day_source=None))
    # A holiday or unknown previous session is a null prior-day R, never an exception.
    odd=prior_day_inputs(NS(mode='live',markets=c.markets),'2026-09-27',runs)
    assert all(v['r'] is None and 'previous session unknown' in v['note'] for v in odd.values())

def test_record_day_r_never_raises_or_downgrades(tmp_path):
    from types import SimpleNamespace as NS
    from open_breakout.standby import record_day_r
    cfg=NS(markets=[NS(name='NQ'),NS(name='ES')])
    write_journal(tmp_path/'trades.sqlite',[trade('NQ',1,-1,[(1,100.)],110.,[('STOP',1,110.5)])])
    runtime=Store(tmp_path/'runtime.sqlite','fp',PREV);ledger=Store(tmp_path/'trades.sqlite','fp',PREV)
    try:
        record_day_r(runtime,ledger,cfg)
        good=runtime.get('day_r');assert good['NQ']['r']==pytest.approx(-1.05) and good['ES']['r'] is None
        # A second call without a journal (restart that failed before opening it) writes nothing.
        record_day_r(runtime,None,cfg);assert runtime.get('day_r')==good
        # A failing journal read records the error and keeps the good value.
        record_day_r(runtime,NS(day_r=lambda names:1/0),cfg)
        assert runtime.get('day_r')==good and 'ZeroDivisionError' in runtime.get('day_r_error')
        # A null never replaces a recorded non-null R.
        record_day_r(runtime,NS(day_r=lambda names:{m:dict(r=None,trades=0,trades_r=[],note='x') for m in names}),cfg)
        assert runtime.get('day_r')['NQ']==good['NQ']
        # A closed runtime store cannot make it raise.
        dead=NS(get=lambda *a:1/0,set=lambda *a:1/0);record_day_r(dead,ledger,cfg)
    finally:ledger.close();runtime.close()

OK_INFO=lambda ratio:dict(atr20=10.,ratio=ratio,atr20_window=['a','b'],roll_excluded=[])
PRIOR=lambda r:dict(r=r,source='shadow_runtime_meta' if r is not None else None,note='n')

@pytest.mark.parametrize('ratio,prior_r,skip',[
    (1.5,2.5,True),     # wide prior session and a big win: skip
    (1.25,2.0,True),    # both boundaries inclusive
    (1.5,1.99,False),   # small win: trade
    (1.5,-1.0,False),   # loss: trade
    (1.5,None,False),   # no prior-day R: trade (fail open on this leg)
    (1.0,3.0,False),    # normal range after a big win: trade
    (None,3.0,True),    # ratio UNAVAILABLE: fail closed as before
    (None,None,True),
])
def test_big_win_truth_table_and_consistency(ratio,prior_r,skip):
    filt=RangeFilter(True,1.25,'skip',True,2.)
    info=OK_INFO(ratio) if ratio is not None else 'only 19 valid prior TRs (need 20)'
    d=range_decision(filt,info,PRIOR(prior_r))
    assert d['skip_prior_range'] is skip and d['half_prior_range'] is False
    assert d['prior_day_r']==prior_r and d['prior_big_win']==(None if prior_r is None else prior_r>=2.)
    assert d['prior_range_reason']
    cfg=NS_CONFIG(filt)
    check_range_fields(cfg,d)
    with pytest.raises(ValueError):check_range_fields(cfg,dict(d,skip_prior_range=not skip))
    if prior_r is not None and ratio is not None:
        with pytest.raises(ValueError,match='prior_big_win'):check_range_fields(cfg,dict(d,prior_big_win=not d['prior_big_win']))
    # Shadow (filter absent or disabled) records the same fields and never skips.
    for f in [None,RangeFilter(False,1.25,'skip',True,2.)]:
        s=range_decision(f,info,PRIOR(prior_r))
        assert not s['skip_prior_range'] and s['prior_day_r']==prior_r and s['prior_big_win']==d['prior_big_win']
        check_range_fields(NS_CONFIG(f),s)

def NS_CONFIG(filt):
    from types import SimpleNamespace as NS
    return NS(prior_range_filter=filt)

def test_big_win_leg_off_keeps_the_plain_rule():
    plain=RangeFilter(True,1.25,'skip')
    d=range_decision(plain,OK_INFO(1.5),PRIOR(-1.))
    assert d['skip_prior_range'] and d['prior_range_reason']=='ratio 1.5000 >= threshold 1.25' and d['prior_big_win'] is False
    check_range_fields(NS_CONFIG(plain),d)
    assert not range_decision(plain,OK_INFO(1.2),PRIOR(3.))['skip_prior_range']
    # Missing prior fields (a legacy manifest) are a null prior-day R: no skip under the big-win rule.
    legacy=dict(prior_range_status='OK',ratio=1.5,skip_prior_range=False,half_prior_range=False)
    check_range_fields(NS_CONFIG(RangeFilter(True,1.25,'skip',True,2.)),legacy)
    with pytest.raises(ValueError):check_range_fields(NS_CONFIG(RangeFilter(True,1.25,'skip',True,2.)),dict(legacy,skip_prior_range=True))
    with pytest.raises(ValueError,match='without a prior-day R'):
        check_range_fields(NS_CONFIG(RangeFilter(True,1.25,'skip',True,2.)),dict(legacy,prior_big_win=False))
    # Half mode under the big-win rule halves only after a big win.
    half=RangeFilter(True,1.25,'half',True,2.)
    assert range_decision(half,OK_INFO(1.5),PRIOR(2.5))['half_prior_range'] and not range_decision(half,OK_INFO(1.5),PRIOR(1.))['half_prior_range']

@pytest.mark.parametrize('prior_r,skip',[(2.4,True),(-1.,False),(None,False)])
def test_manifest_big_win_rule_roundtrip(config,tmp_path,prior_r,skip):
    c=filtered(tmp_path,{**BIG,'threshold':1.2},'big.json')
    risk,bars,rb=range_manifest_inputs(c)
    prior={'NQ':PRIOR(prior_r),'ES':PRIOR(None)}
    b=build_manifest(c,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=rb,prior_day=prior)
    assert b['prior_range_filter']=={**BIG,'threshold':1.2}
    nq,es=b['markets']['NQ'],b['markets']['ES']
    assert nq['ratio']==pytest.approx(40/32.4) and nq['skip_prior_range'] is skip and nq['prior_day_r']==prior_r
    assert es['skip_prior_range'] is False and es['prior_big_win'] is None and 'fail open' in es['prior_range_reason']
    p=tmp_path/'inputs.json';p.write_text(json.dumps(b));assert load_manifest(p,c)==b
    forged={k:dict(v) for k,v in b['markets'].items()};forged['ES']['skip_prior_range']=True
    with pytest.raises(ValueError,match='prior-day R'):check_range_fields(c,forged['ES'])
    # The same prior-day inputs under the plain rule skip both wide markets.
    plain=filtered(tmp_path,{**FILTER,'threshold':1.2},'plain.json')
    assert all(v['skip_prior_range'] for v in build_manifest(plain,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,
                                                             range_bars=rb,prior_day=prior)['markets'].values())
    assert build_manifest(plain,DAY,risk,bars,now=at('09:00:00'),roll_verified=True,range_bars=rb)['prior_range_filter']=={**FILTER,'threshold':1.2}

def test_big_win_skip_event_carries_ratio_prior_day_r_and_thresholds(config,tmp_path):
    c=filtered(tmp_path,BIG,'big.json')
    now=[at('09:29:00')]
    store=Store(tmp_path/'s.sqlite',c.fingerprint,DAY)
    try:
        m=skip_manifest(c,'ES',prior_day_r=2.6,prior_big_win=True,prior_day_source='shadow_runtime_meta')
        service=Service(c,m,store,SimBroker(c,lambda:now[0]),lambda:now[0])
        assert service.states['ES'].phase=='SKIPPED' and service.states['NQ'].phase=='FLAT'
        ev=[json.loads(v) for k,v in store.db.execute('SELECT kind,body FROM events') if k=='PRIOR_RANGE_SKIP']
        assert len(ev)==1 and ev[0]['ratio']==pytest.approx(40/28.) and ev[0]['prior_day_r']==2.6
        assert ev[0]['threshold']==1.25 and ev[0]['big_win_r']==2. and ev[0]['require_prior_big_win'] is True
    finally:store.close()
    from open_breakout.standby import range_line
    line=range_line(dict(m,prior_range_filter=BIG))
    assert range_line(m).endswith('filter off')
    assert 'ES ratio 1.429 prior-day R +2.60' in line and 'NQ ratio 1.000 prior-day R n/a' in line
    assert 'threshold 1.25 and big win 2.0R' in line

def test_prepare_cli_and_status_print_prior_day_fields(config,tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace as NS
    import open_breakout.ibkr as ibkr_module
    import open_breakout.__main__ as cli
    c=filtered(tmp_path,{**BIG,'threshold':1.2},'big.json')
    risk,bars,rb=range_manifest_inputs(c)
    rpath=tmp_path/'risk.parquet';risk.to_parquet(rpath)
    runs=tmp_path/'runs'
    write_journal(runs/f'{PREV}-shadow'/'trades.sqlite',[trade('NQ',1,1,[(1,100.)],90.,[('TIME',1,130.)])])
    class Fake:
        def __init__(self,config,session=None):
            self.ib=NS(client=NS(placeOrder=lambda *a:None),isConnected=lambda:True)
        async def connect(self):pass
        async def history(self):return bars
        async def range_history(self):return rb
        def close(self):pass
    monkeypatch.setattr(ibkr_module,'IBKR',Fake)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return at('09:00:00').astimezone(tz) if tz else at('09:00:00')
    monkeypatch.setattr(cli,'datetime',Clock)
    out=runs/f'{DAY}-shadow'/'inputs.json'
    asyncio.run(cli.main_async(NS(command='prepare',config=str(tmp_path/'big.json'),session=DAY,
        risk_parquet=str(rpath),out=str(out),roll_verified=True,client_id=None,runs_root=str(runs))))
    text=capsys.readouterr().out
    printed=json.loads(text[:text.rindex('}')+1])
    nq,es=printed['markets']['NQ'],printed['markets']['ES']
    assert nq['prior_day_r']==pytest.approx(3.) and nq['prior_big_win'] is True and nq['skip_prior_range'] is True
    assert nq['prior_day_source']=='shadow_journal' and es['prior_day_r'] is None and es['skip_prior_range'] is False
    Store(out.parent/'runtime.sqlite','fp',DAY).close()
    asyncio.run(cli.main_async(NS(command='status',state=str(out.parent/'runtime.sqlite'))))
    status=json.loads(capsys.readouterr().out)
    assert status['prior_range']['markets']['NQ']['prior_day_r']==pytest.approx(3.)
    assert status['prior_range']['filter']['require_prior_big_win'] is True
