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

async def tick(service,broker,now,price,stamp,market='NQ'):
    now[0]=at(stamp)
    broker.update_quote(market,price-.25,price,now[0])
    service.tick(market,now[0],price)
    await service.drain()


def test_config_boundaries(config,tmp_path,monkeypatch):
    config.authorize()
    paper=replace(config,mode='paper');paper.authorize()
    with pytest.raises(PermissionError):replace(paper,account='U_REAL').authorize()
    with pytest.raises(PermissionError):replace(paper,port=7496).authorize()
    live=replace(config,mode='live',account='U_REAL',allow_live=True)
    with pytest.raises(PermissionError):live.authorize(DAY)
    # A replaced config without the pilot block/one-lot caps is never live-authorizable.
    monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} U_REAL')
    with pytest.raises(PermissionError,match='pilot'):live.authorize(DAY)
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
            self.positions[cid]=self.positions.get(cid,0)+o['body']['side']*q
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

def test_foreign_position_blocks_and_lock_excludes_second_process(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            with pytest.raises(RuntimeError):Store(tmp_path/'session.sqlite',config.fingerprint,DAY)
            b.positions[config.markets[0].execution.con_id]=1
            # One discrepancy may be a fill in flight; three consecutive stable checks halt.
            await service.watchdog();await service.watchdog();assert not service.halted
            await service.watchdog();assert service.halted
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
            store.set('daily_reserved',0.)
            service.states['ES'].planned_risk=100.
            await tick(service,b,now,20000,'09:30:02');await tick(service,b,now,20010,'09:30:03')
            assert not b.orders and service.states['NQ'].attempts==0
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

def test_foreign_client_order_blocks(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        try:
            async def snapshot():
                return dict(positions={},orders=[dict(id=1,client_id=999,con_id=config.markets[0].execution.con_id,ref='OTHER')])
            b.snapshot=snapshot
            await service.watchdog();assert service.halted
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
        monkeypatch.setenv('OPEN_BREAKOUT_LIVE_ACK',f'LIVE {DAY} {live.account}')
        with pytest.raises(PermissionError,match='pilot'):await IBKR(live,session=DAY).connect()
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


# ---- Live one-contract pilot ----

LIVE_ACCOUNT='U9990001'

def live_raw(**changes):
    raw=json.loads((ROOT/'config/open_breakout.example.json').read_text(encoding='utf-8-sig'))
    raw.update(mode='live',allow_live=True,account=LIVE_ACCOUNT,port=7496,client_id=927999,
               pilot={'max_contracts_per_market':1})
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
    from open_breakout.config import LIVE_PILOT_MAX_CONTRACTS
    assert LIVE_PILOT_MAX_CONTRACTS==1
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
    lambda r:r['markets'][0].update(max_contracts=2),
    lambda r:r['markets'][1].update(max_contracts=2),
    lambda r:r.update(account='DU1234567'),
    lambda r:r.pop('pilot'),
    lambda r:r.update(pilot={'max_contracts_per_market':2}),
    lambda r:r.update(pilot={'max_contracts_per_market':1,'extra':1}),
    lambda r:r.update(allow_live=False),
])
def test_live_config_structural_rejections(tmp_path,mutate):
    raw=live_raw();mutate(raw)
    with pytest.raises((PermissionError,ValueError)):load_raw(tmp_path,raw)

def test_shadow_fingerprint_unchanged_without_pilot(config,tmp_path):
    raw=json.loads((tmp_path/'config.json').read_text())
    assert 'pilot' not in raw and Config.load(tmp_path/'config.json').fingerprint==config.fingerprint
    assert config.pilot_max_contracts==0 and config.authorize() is None

def test_live_sizing_clamps_to_one_and_zero_stays_zero(config):
    s=State(DAY,'NQ',40.1,25)
    shadow=size_order(s,config.markets[0],1,20000,20000.25,100000,config)
    assert shadow['qty']==6 # live clamp is not applied in shadow
    live_cfg=replace(config,mode='live')
    p=size_order(s,config.markets[0],1,20000,20000.25,100000,live_cfg)
    assert p['qty']==1 and p['risk']==pytest.approx(shadow['risk']/6)
    assert size_order(s,config.markets[0],1,20000,20000.25,1000,live_cfg)['qty']==0

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

def test_broker_position_mismatch_skips_flatten(config,tmp_path):
    async def run():
        service,b,store,now=setup_service(config,tmp_path)
        service.flatten_delay=0
        try:
            await open_nq(service,b,now)
            s=service.states['NQ']
            b.positions[config.markets[0].execution.con_id]=0
            b.order_error_callback(s.stop_order,201,'Order rejected')
            await service.drain()
            assert not flatten_orders(b) and service.halted
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

def test_preflight_reports_foreign_orders_positions_and_margin(live):
    from types import SimpleNamespace as NS
    from open_breakout.standby import preflight
    exec_ids=[m.execution.con_id for m in live.markets]
    class Fake:
        config=live;ack='ack';healthy=True;contract_report={'MNQ':{'ok':True}};data_types={1:1}
        def __init__(self,positions,orders,margin):
            now=datetime.now(NY)
            self.last_seen={f'{m.name}:{k}':now for m in live.markets for k in ('trade','quote')}
            self.margin=margin
            async def pos():return positions
            async def orders_():return orders
            self.ib=NS(reqPositionsAsync=pos,reqAllOpenOrdersAsync=orders_)
        async def account_values(self):return {'NetLiquidation':600000.,'ExcessLiquidity':500000.}
        async def what_if(self,m,body):
            assert body['qty']==1 and body['kind']=='MKT'
            return NS(initMarginChange=str(self.margin),maintMarginChange='1',commission=0.,warningText='')
    async def run():
        clean=await preflight(Fake([],[],3000.),wait_seconds=0)
        assert clean['ok'] and set(clean['what_if'])=={'MNQ:BUY','MNQ:SELL','MES:BUY','MES:SELL'}
        pos=[NS(account=live.account,contract=NS(conId=exec_ids[1]),position=1.)]
        foreign=[NS(contract=NS(conId=exec_ids[0]),order=NS(clientId=0,orderId=5,permId=9,orderType='STP',action='SELL',
                    totalQuantity=1,orderRef='manual'),orderStatus=NS(status='PreSubmitted'))]
        bad=await preflight(Fake(pos,foreign,200000.),wait_seconds=0)
        assert not bad['ok']
        text=' '.join(bad['failures'])
        assert 'NONZERO_POSITION:MES' in text and 'WORKING_ORDER:MNQ:client0' in text and 'MARGIN:MNQ:BUY' in text
    asyncio.run(run())

def test_live_session_arms_after_preflight_and_trades_one_lot(live,tmp_path,monkeypatch):
    import sqlite3
    from types import SimpleNamespace as NS
    from open_breakout import standby
    from open_breakout.__main__ import main_async
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
        async def what_if(self,m,body):return NS(initMarginChange='3000',maintMarginChange='2000',commission=0.,warningText='')
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
        if '09:30:00'<=stamp<'16:01:00':
            for m,base in [('NQ',20000),('ES',5000)]:
                price=base+(10 if stamp>'09:30:00' else 0)
                feed.update_quote(m,price-.25,price,clock[0])
                feed.capture(dict(kind='quote',market=m,time=clock[0].isoformat(),bid=price-.25,ask=price))
                feed.on_tick(m,clock[0],price)
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
    # NQ sizes to 6 micros unclamped -> 1; ES budget (5bp of 100k) is below one MES of risk -> no trade.
    assert len(entries)==1 and entries[0]['qty']==1 and entries[0]['ref'].startswith('MNQ|BUY|OpenBreakout|')
    assert [role for role,_ in orders]==['ENTRY','STOP','TIME']
    assert len(placed)==len(orders) and all(b['account']==LIVE_ACCOUNT for _,b in orders)


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
        adapter.ib=NS(reqAllOpenOrdersAsync=lambda:slow([]),reqPositionsAsync=lambda:slow([]))
        await asyncio.gather(adapter.snapshot(),adapter.snapshot(),adapter.snapshot())
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
