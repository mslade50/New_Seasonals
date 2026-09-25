"""Commands: validate, prepare, run (shadow/paper), shadow-session, live-session, preflight (read only),
replay, status, reconcile (read only)."""
import argparse
from dataclasses import replace
import gzip
import asyncio
from datetime import datetime, timezone, time
import json
from pathlib import Path
import sqlite3
from .config import Config
from .inputs import build_manifest, load_manifest
from .strategy import NY, aware
from .store import Store
from .service import Service
from .replay import SimBroker


def write_new(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(body,f,indent=2)

async def main_async(args):
    if args.command=='status':
        # SQLite read-only URI; never creates a missing database or acquires its writer lock.
        path=Path(args.state).resolve()
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            print(json.dumps(dict(meta={k:json.loads(v) for k,v in db.execute('SELECT * FROM meta')},
                states=[json.loads(x[0]) for x in db.execute('SELECT body FROM states')]),indent=2))
        return
    config=Config.load(args.config)
    session=getattr(args,'session',None)
    if getattr(args,'client_id',None):
        # Read-only commands while a session process owns the configured client ID.
        if args.client_id==config.client_id:raise ValueError('--client-id must differ from the configured session client ID')
        config=replace(config,client_id=args.client_id)
    if args.command=='shadow-session':
        from .standby import run_shadow
        await run_shadow(args.config,args.session,args.risk_parquet,args.state_dir,args.roll_verified)
        return
    if args.command=='live-session':
        if config.mode!='live':
            raise PermissionError('live-session requires a live pilot configuration')
        # Process-start acknowledgement check; repeated inside IBKR.connect().
        config.authorize(args.session)
        from .standby import run_live
        await run_live(args.config,args.session,args.risk_parquet,args.state_dir,args.roll_verified)
        return
    if args.command=='validate':
        ack=config.authorize(session)
        print(json.dumps(dict(mode=config.mode,config_hash=config.fingerprint,live_release_enabled=config.mode=='live',
                              live_ack_verified=ack is not None,markets=[m.name for m in config.markets])))
        return
    if args.command=='preflight':
        from .ibkr import IBKR
        from .standby import OrderGate, preflight
        transport=IBKR(config,session=session)
        # Read-only: only what-if previews may reach the client; the gate is never opened.
        OrderGate(transport.ib.client)
        try:
            await asyncio.wait_for(transport.connect(),30)
            transport.subscribe(lambda *a:None,lambda e:None)
            report=await preflight(transport,wait_seconds=args.wait)
            report['orders_placed']=0
        finally:
            transport.close()
        if args.out:write_new(args.out,report)
        print(json.dumps(report,indent=2,default=str))
        return
    if args.command=='replay':
        manifest=load_manifest(args.inputs,config)
        now=[aware(manifest['prepared_at'])]
        broker=SimBroker(config,lambda:now[0])
        with_store=Store(args.state,config.fingerprint,manifest['day'])
        try:
            service=Service(config,manifest,with_store,broker,lambda:now[0])
            tick_path=Path(args.ticks)
            opener=gzip.open if tick_path.suffix=='.gz' else open
            with opener(tick_path,'rt',encoding='utf-8-sig') as f:
                for line in f:
                    e=json.loads(line);now[0]=aware(e.get('received_at',e['time']))
                    timestamp=aware(e['time'])
                    if e['kind']=='quote':broker.update_quote(e['market'],e['bid'],e['ask'],timestamp)
                    elif e['kind']=='trade':service.tick(e['market'],timestamp,e['price'])
                    else:raise ValueError('Unknown replay event type')
                    await service.drain()
                    await service.watchdog()
            print(json.dumps(dict(halted=service.halted,states=[s.record() for s in service.states.values()]),indent=2))
        finally:with_store.close()
        return
    from .ibkr import IBKR
    if config.mode=='live' and args.command=='run':
        raise PermissionError('Live routing is available only through live-session')
    transport=IBKR(config,session=session)
    if args.command!='run':
        from .standby import OrderGate
        OrderGate(transport.ib.client)
    store=None
    try:
        await transport.connect()
        if args.command=='reconcile':
            snapshot=await transport.snapshot()
            from ib_insync import ExecutionFilter
            fills=await transport.ib.reqExecutionsAsync(ExecutionFilter(acctCode=config.account))
            path=Path(args.state).resolve()
            with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
                meta={k:json.loads(v) for k,v in db.execute('SELECT * FROM meta')}
                if meta['fingerprint']!=config.fingerprint:raise ValueError('Journal/config mismatch')
                states=[json.loads(r[0]) for r in db.execute('SELECT body FROM states')]
                journal_orders=[dict(id=r[0],market=r[1],role=r[2],status=r[3]) for r in db.execute('SELECT * FROM orders')]
            report=dict(broker=snapshot,journal_states=states,journal_orders=journal_orders,
                        executions=[dict(id=f.execution.execId,order_id=f.execution.orderId,
                            client_id=f.execution.clientId,con_id=f.contract.conId,qty=f.execution.shares,
                            price=f.execution.price) for f in fills],
                        action='Inspection only. Halt remains set; no resubmission or position changes.')
            print(json.dumps(report,indent=2))
            return
        if args.command=='prepare':
            import pandas as pd
            bars=await transport.history()
            body=build_manifest(config,args.session,pd.read_parquet(args.risk_parquet),bars,
                                now=datetime.now(timezone.utc),roll_verified=args.roll_verified)
            write_new(args.out,body)
            print(f'Prepared {args.out}; no orders submitted')
            return
        manifest=load_manifest(args.inputs,config)
        now=datetime.now(timezone.utc)
        if now.astimezone(NY).date().isoformat()!=manifest['day']:
            raise ValueError('Manifest is not for today')
        if now.astimezone(NY).time()>=time(9,30):
            raise ValueError('Start before 09:30 ET; late starts require manual reconciliation, not new entries')
        store=Store(args.state,config.fingerprint,manifest['day'])
        broker=SimBroker(config,lambda:datetime.now(timezone.utc)) if config.mode=='shadow' else transport
        service=Service(config,manifest,store,broker)
        # Explicitly refresh open orders and account positions before any subscription.
        await service.watchdog()
        if service.halted:raise RuntimeError('Startup reconciliation halted; inspect status journal')
        if config.mode=='shadow':transport.halt_callback=service.halt
        capture_path=Path(args.capture).resolve()
        if any(p.lower().startswith('onedrive') for p in capture_path.parts):raise ValueError('Capture must be outside OneDrive')
        capture_path.parent.mkdir(parents=True,exist_ok=True)
        with capture_path.open('a',encoding='utf-8') as capture_file:
            def capture(event):
                event['received_at']=datetime.now(timezone.utc).isoformat()
                capture_file.write(json.dumps(event)+'\n');capture_file.flush()
                if config.mode=='shadow' and event['kind']=='quote':
                    broker.update_quote(event['market'],event['bid'],event['ask'],event['time'])
            transport.subscribe(service.tick,capture)
            reported=False
            reported_markets=set()
            while transport.ib.isConnected():
                await asyncio.sleep(1)
                await service.watchdog()
                if service.halted and not reported:
                    print('HALTED: inspect journal and broker; existing protective orders retained',flush=True);reported=True
                for name,state in service.states.items():
                    if state.phase=='BLOCKED' and name not in reported_markets:
                        print(f'{name} skipped: {state.note}',flush=True);reported_markets.add(name)
                local=datetime.now(timezone.utc).astimezone(NY)
                if local.time()>=time(16,1):break
            await service.drain()
    finally:
        transport.close()
        if store:store.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['validate','prepare','run','replay','reconcile']:
        q=sub.add_parser(name);q.add_argument('--config',required=True)
        if name in {'run','replay'}:
            q.add_argument('--inputs',required=True);q.add_argument('--state',required=True)
        if name=='reconcile':
            q.add_argument('--state',required=True);q.add_argument('--session',help='required for a live config')
            q.add_argument('--client-id',type=int,help='read-only client ID while the session process is alive (e.g. 927482)')
        if name=='validate':q.add_argument('--session',help='required for a live config')
        if name=='prepare':
            q.add_argument('--session',required=True);q.add_argument('--risk-parquet',required=True)
            q.add_argument('--out',required=True);q.add_argument('--roll-verified',action='store_true')
        if name=='run':q.add_argument('--capture',required=True)
        if name=='replay':q.add_argument('--ticks',required=True)
    for name in ['shadow-session','live-session']:
        q=sub.add_parser(name)
        q.add_argument('--config',required=True);q.add_argument('--session',required=True)
        q.add_argument('--risk-parquet',required=True);q.add_argument('--state-dir',required=True)
        q.add_argument('--roll-verified',action='store_true')
    q=sub.add_parser('preflight',help='read-only broker checks; places no orders')
    q.add_argument('--config',required=True);q.add_argument('--session',required=True)
    q.add_argument('--wait',type=float,default=30.);q.add_argument('--out')
    q.add_argument('--client-id',type=int,help='read-only client ID while the session process is alive (e.g. 927482)')
    q=sub.add_parser('status');q.add_argument('--state',required=True)
    asyncio.run(main_async(p.parse_args()))

if __name__=='__main__':main()
