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
        body=dict(kind='MKT',side=-1,qty=2,tif='GTC',oca='test-group',account=paper.account,ref='OB:test:TIME',good_after='20260924 15:55:00 US/Eastern')
        adapter.send(18,m,body)
        assert sent[1].goodAfterTime.endswith('US/Eastern') and sent[1].ocaGroup==sent[0].ocaGroup
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
        assert tx.orderType=='MKT' and tx.tif=='GTC' and tx.goodAfterTime==f'{DAY.replace("-","")} 15:55:00 US/Eastern'
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
            assert store.orders()[s.time_order]['body']['good_after'].endswith('15:55:00 US/Eastern')
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
