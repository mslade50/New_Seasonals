"""One-session runners. Shadow: record, prepare before open, simulate only.
Live: same schedule, read-only preflight at connect and 09:25 ET, broker orders only after arming."""
import asyncio
from datetime import datetime, time, timezone, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
import os
import sys
import pandas as pd
from .alerts import Alerts, webhook_url
from .config import Config, LIVE_HARD_MAX_CONTRACTS
from .ibkr import IBKR
from .inputs import build_manifest, calendar_dates
from .replay import SimBroker
from .service import Service, own_ref, prior_range_skip
from .store import Store, prior_day_r
from .strategy import NY, State, size_order

# Connect-time (pre-manifest) what-if size, warning only. The prior TR needs the 09:00 history pull, so the
# ~08:15 preflight previews a fixed reference size (clamped to the effective cap); the 09:25 planned-size
# what-if is the gate.
REFERENCE_WHATIF_CONTRACTS = 10


async def fetch_range_bars(transport):
    """Prior-range history; a failure is recorded in the manifest (fail closed per market), not raised."""
    try:
        return await asyncio.wait_for(transport.range_history(),450),None
    except Exception as exc:
        return None,f'{type(exc).__name__}: {exc}'


RANGE_FIELDS=('prior_range_status','atr20','ratio','prior_day_r','prior_day_source','prior_big_win',
              'skip_prior_range','half_prior_range','prior_range_reason')


def range_summary(manifest):
    return {k:{f:v.get(f) for f in RANGE_FIELDS} for k,v in manifest['markets'].items()}


def range_line(manifest):
    """One line for alerts: ratio and prior-day R per market, and the thresholds in force."""
    num=lambda x,fmt:'n/a' if x is None else format(x,fmt)
    parts=[f'{k} ratio {num(v.get("ratio"),".3f")} prior-day R {num(v.get("prior_day_r"),"+.2f")}'
           +(f' ({v["prior_day_source"]})' if v.get('prior_day_source') else '')
           for k,v in manifest['markets'].items()]
    filt=manifest.get('prior_range_filter')
    if filt is None:
        rule='filter off'
    else:
        rule=(f'threshold {filt["threshold"]}'+(f' and big win {filt["big_win_r"]}R' if filt.get('require_prior_big_win') else '')
              +('' if filt['enabled'] else ' (disabled)'))
    return f'range [{"; ".join(parts)}] {rule}'


def prior_day_inputs(config,day,runs_root):
    """Prior-day R per market for the previous XNYS session: the unfiltered shadow journal
    (<runs_root>/<previous>-shadow[-N]) first, then, in live mode, the live journal (-live[-N]).
    Never raises into the 09:00 prepare: any failure is a null prior-day R (no skip)."""
    names=[m.name for m in config.markets]
    try:
        previous,_=calendar_dates(day)
    except Exception as exc:
        return {m:dict(r=None,source=None,note=f'previous session unknown ({type(exc).__name__}: {exc})') for m in names}
    return prior_day_r(Path(runs_root),previous,'live' if config.mode=='live' else 'shadow',names)


def record_day_r(runtime,ledger,config):
    """Session's own R per market into the runtime meta, read by the next session's prepare. Never raises.
    Without an open journal nothing is written (the reader recomputes from the journal file, if any), and a
    market's non-null R already recorded is never replaced by a null one."""
    try:
        if not ledger:
            return
        names=[m.name for m in config.markets]
        summary=ledger.day_r(names)
        old=runtime.get('day_r')
        if isinstance(old,dict):
            for m in names:
                if summary[m].get('r') is None and isinstance(old.get(m),dict) and old[m].get('r') is not None:
                    summary[m]=old[m]
        runtime.set('day_r',summary)
    except Exception as exc:
        try:runtime.set('day_r_error',f'{type(exc).__name__}: {exc}')
        except Exception:pass


def session_bounds(day):
    calendar_dates(day)
    d=datetime.fromisoformat(day).date()
    return (datetime.combine(d,time(9),NY),datetime.combine(d,time(9,30),NY),
            datetime.combine(d,time(16,1),NY))


def validate_launch(config,day,now):
    if config.mode!='shadow' or config.allow_live:
        raise PermissionError('The standby runner supports shadow mode only')
    prepare,opening,closing=session_bounds(day)
    if now>=opening or now<opening-timedelta(days=1):
        raise ValueError('Start within 24 hours before the requested cash open')
    return prepare,opening,closing


def validate_live_launch(config,day,now):
    if config.mode!='live' or not config.allow_live:
        raise PermissionError('live-session requires a live configuration')
    ack=config.authorize(day)
    prepare,opening,closing=session_bounds(day)
    if now>=opening or now<opening-timedelta(days=1):
        raise ValueError('Start within 24 hours before the requested cash open')
    return prepare,opening,closing,ack


class OrderGate:
    """Below the adapter: only what-if previews pass until the live session is armed."""
    def __init__(self,client):
        self.client=client;self.original=client.placeOrder;self.open=False
        client.placeOrder=self.place

    def place(self,order_id,contract,order):
        if not self.open and not getattr(order,'whatIf',False):
            raise PermissionError('Live orders are blocked until preflight passes and the session is armed')
        return self.original(order_id,contract,order)


WORKING_DONE={'Filled','Cancelled','ApiCancelled','Inactive'}


async def book_view(transport):
    """Read-only view of the execution contracts, split by orderRef. OpenBreakout's own position comes from
    its own executions; the account position and other strategies' working orders are other_book."""
    config=transport.config
    execution={m.execution.con_id:m.execution.symbol for m in config.markets}
    lock=getattr(transport,'snapshot_lock',None) or asyncio.Lock()
    async with lock:
        positions=await asyncio.wait_for(transport.ib.reqPositionsAsync(),10)
        trades=await asyncio.wait_for(transport.ib.reqAllOpenOrdersAsync(),10)
    own=None;own_error=None
    for attempt in range(3):
        try:
            own={symbol:await asyncio.wait_for(transport.own_position(cid),15) for cid,symbol in execution.items()}
            own_error=None;break
        except Exception as exc:
            # Executions unreadable: our own position is UNKNOWN, never assumed flat.
            own=None;own_error=f'{type(exc).__name__}: {exc}'
            if attempt<2:await asyncio.sleep(1)
    held={symbol:sum(p.position for p in positions if p.account==config.account and p.contract.conId==cid)
          for cid,symbol in execution.items()}
    working=[dict(symbol=execution[t.contract.conId],client_id=t.order.clientId,order_id=t.order.orderId,
                  perm_id=t.order.permId,type=t.order.orderType,action=t.order.action,qty=t.order.totalQuantity,
                  status=t.orderStatus.status,ref=t.order.orderRef)
             for t in trades if t.contract.conId in execution and t.orderStatus.status not in WORKING_DONE]
    other={s:dict(account_position=held[s],other_position=None if own is None else held[s]-own[s],
                  working_orders=[w for w in working if w['symbol']==s and not own_ref(w['ref'])]) for s in held}
    return dict(own_positions=own,own_positions_error=own_error,working_orders=[w for w in working if own_ref(w['ref'])],
                other_book=other,open_orders_seen_any_contract=len(trades))


async def preflight(transport,*,manifest=None,wait_seconds=30.,max_age=15.):
    """Read-only checks: streams, OpenBreakout's own position and working orders (all clients), contracts,
    account and what-if margin. Other strategies in the same contracts are reported as other_book only.
    Margin: one contract each side always; then the planned sizes when the manifest exists (fails), else
    a fixed reference size (warnings only)."""
    config=transport.config
    failures=[];warnings=[]
    report=dict(checked_at=datetime.now(timezone.utc).isoformat(),mode=config.mode,account_suffix=config.account[-4:],
                port=config.port,client_id=config.client_id,live_ack_verified=transport.ack is not None,
                contracts=transport.contract_report,failures=failures,warnings=warnings)
    needed={f'{m.name}:{k}' for m in config.markets for k in ('trade','quote')}
    deadline=asyncio.get_running_loop().time()+wait_seconds
    while not needed<=set(transport.last_seen) and asyncio.get_running_loop().time()<deadline:
        await asyncio.sleep(.25)
    now=datetime.now(timezone.utc)
    ages={k:round((now-transport.last_seen[k]).total_seconds(),3) for k in sorted(needed) if k in transport.last_seen}
    report['stream_age_seconds']=ages
    for k in sorted(needed):
        if k not in ages:failures.append(f'STREAM_MISSING:{k}')
        elif ages[k]>max_age:failures.append(f'STREAM_STALE:{k}:{ages[k]}s')
    report['market_data_types']={str(k):v for k,v in transport.data_types.items()}
    if any(v!=1 for v in transport.data_types.values()):failures.append('NON_LIVE_MARKET_DATA')
    if not transport.healthy:failures.append('TRANSPORT_UNHEALTHY')
    view=await book_view(transport)
    report.update(view)
    # Ownership is the orderRef: only OpenBreakout's own position and working orders fail preflight.
    if view['own_positions'] is None:
        # Fail closed before any order: flat cannot be verified without our executions.
        failures.append(f'OWN_POSITION_UNKNOWN:executions request failed ({view["own_positions_error"]})')
    else:
        failures.extend(f'OWN_POSITION:{s}:{q}' for s,q in view['own_positions'].items() if q)
    failures.extend(f'WORKING_ORDER:{w["symbol"]}:client{w["client_id"]}:{w["type"]}' for w in view['working_orders'])
    try:
        values=await asyncio.wait_for(transport.account_values(),15)
        report['net_liquidation']=values['NetLiquidation'];report['excess_liquidity']=values['ExcessLiquidity']
        limit=min(values['ExcessLiquidity'],values['NetLiquidation'])*config.max_margin_fraction
        report['margin_limit']=limit
    except Exception as exc:
        failures.append(f'ACCOUNT_VALUES:{exc}');limit=float('nan')
    async def preview(m,side,qty,problems,tag,within_limit=True):
        """One MKT what-if; returns (row, initial margin change or None when unusable). Sized previews at
        the planned basis pass within_limit=False: over the limit there is a size-down, not a failure."""
        key=f'{m.execution.symbol}:{"BUY" if side==1 else "SELL"}'
        try:
            r=await transport.what_if(m,dict(kind='MKT',side=side,qty=qty,tif='DAY',ref='OB:WHATIF'))
            change=float(r.initMarginChange)
            row=dict(qty=qty,init_margin_change=change,maint_margin_change=float(r.maintMarginChange),
                     commission=r.commission,warning=r.warningText or '')
            if not math.isfinite(change) or change<0 or (within_limit and not change<=limit) or r.warningText:
                problems.append(f'{tag}:{key}:{change}');return row,None
            return row,change
        except Exception as exc:
            problems.append(f'{tag}:{key}:{exc}');return dict(qty=qty,error=str(exc)),None
    async def sized_check(qtys,problems,tag,within_limit):
        """What-if each side at qtys[sym][action]; returns (rows, worse side per market, summed total or None)."""
        rows={};worst={};complete=True
        for m in config.markets:
            sym=m.execution.symbol
            for side in (1,-1):
                action='BUY' if side==1 else 'SELL'
                qty=qtys[sym][action]
                if qty<1:
                    rows[f'{sym}:{action}']=dict(qty=0,init_margin_change=0.);worst.setdefault(sym,0.);continue
                row,change=await preview(m,side,qty,problems,tag,within_limit)
                rows[f'{sym}:{action}']=row
                if change is None:complete=False
                else:worst[sym]=max(worst.get(sym,0.),change)
        total=sum(worst.values()) if complete and len(worst)==len(config.markets) else None
        return rows,worst,total
    # Basic check (contract tradeable, no warning, one lot fits): one contract each side, always failing.
    one={};one_worst={}
    for m in config.markets:
        for side in (1,-1):
            row,change=await preview(m,side,1,failures,'MARGIN')
            one[f'{m.execution.symbol}:{"BUY" if side==1 else "SELL"}']=row
            if change is not None:one_worst[m.execution.symbol]=max(one_worst.get(m.execution.symbol,0.),change)
    report['what_if_one_contract']=one
    # Sized check. With the manifest (09:25 arming, or a reconnect after 09:00): the PLANNED sizes that
    # size_order would send now; when their summed worse side exceeds the limit, a per-market margin day cap
    # sizes the session down to fit (owner decision 2026-09-28; it no longer fails the session). Without it
    # (connect at ~08:15, the launcher's read-only preflight): at REFERENCE_WHATIF_CONTRACTS (clamped to the
    # effective cap), as a warning only. The fat-finger cap (60) is not a meaningful size to preview.
    planned=None
    if manifest is not None:
        try:
            values_equity=config.shadow_equity if config.mode=='live' else report['net_liquidation']
            planned=planned_sizes(config,manifest,transport.quote,values_equity)
        except Exception as exc:
            failures.append(f'PLANNED_SIZE:{type(exc).__name__}: {exc}')
    basis='planned' if planned is not None else 'reference'
    problems=failures if basis=='planned' else warnings
    tag='MARGIN_PLANNED' if basis=='planned' else 'MARGIN_AT_REFERENCE'
    if planned is not None:
        qtys={s:{k:planned[s][k] for k in ('BUY','SELL')} for s in planned}
    else:
        qtys={m.execution.symbol:dict.fromkeys(('BUY','SELL'),min(REFERENCE_WHATIF_CONTRACTS,config.max_contracts_for(m)))
              for m in config.markets}
    sized,worst,total=await sized_check(qtys,problems,tag,basis!='planned')
    report['what_if_basis']=basis
    report['what_if']=sized
    report['what_if_qty']={m.execution.symbol:{k:sized[f'{m.execution.symbol}:{k}']['qty'] for k in ('BUY','SELL')} for m in config.markets}
    report['max_contracts']={m.execution.symbol:config.max_contracts_for(m) for m in config.markets}
    report['margin_day_cap']=None
    if planned is not None:
        report['planned']=planned
        report['planned_qty']={s:max(v['BUY'],v['SELL']) for s,v in planned.items()}
        report['margin_total_planned']=total
    # None when a market's sized what-if was unusable (listed above).
    report['margin_total']=total
    if total is not None and not total<=limit:
        if basis=='planned':
            await apply_margin_day_cap(report,worst,one_worst,limit,sized_check,failures)
        else:
            warnings.append(f'MARGIN_TOTAL_AT_REFERENCE:{total:.2f}>{limit:.2f}')
    report['ok']=not failures
    return report


MARGIN_CAP_MAX_REDUCTIONS = 10


def margin_day_caps(planned_qty,worst,one_worst,limit):
    """Per-market whole-contract caps so the summed worse-side margin fits `limit`. The limit is split in
    proportion to each market's planned margin, so every market scales by the same factor limit/total, and
    floored. A market floored to 0 gets 1 if its one-contract what-if fits beside the other markets' capped
    margin (linear estimate), else 0 (not armed). A market planning 0 needs no cap (None)."""
    factor=limit/sum(worst.values())
    caps={s:(min(q,math.floor(q*factor+1e-9)) if q>0 else None) for s,q in planned_qty.items()}
    est=lambda s:caps[s]*worst[s]/planned_qty[s] if caps[s] else 0.
    for s,q in planned_qty.items():
        if caps[s]==0:
            others=sum(est(o) for o in caps if o!=s)
            caps[s]=1 if one_worst.get(s,math.inf)+others<=limit else 0
    return caps


async def apply_margin_day_cap(report,worst,one_worst,limit,sized_check,failures):
    """09:25: size the session down instead of failing it. Caps from margin_day_caps(), then a re-what-if at
    the capped sizes that must fit; if it does not (nonlinear margins), one contract off one market at a time,
    alternating, at most MARGIN_CAP_MAX_REDUCTIONS times, else MARGIN_TOTAL fails closed as before."""
    planned=report['planned'];planned_qty=report['planned_qty']
    caps=margin_day_caps(planned_qty,worst,one_worst,limit)
    order=list(planned_qty)
    reductions=0;turn=0;total=None
    while True:
        qtys={s:{k:(planned[s][k] if caps[s] is None else min(planned[s][k],caps[s])) for k in ('BUY','SELL')} for s in order}
        rows,_,total=await sized_check(qtys,failures,'MARGIN_CAPPED',False)
        if total is None or total<=limit:
            break
        reducible=[s for s in order if caps[s] is not None and caps[s]>1]
        if not reducible or reductions>=MARGIN_CAP_MAX_REDUCTIONS:
            failures.append(f'MARGIN_TOTAL:{total:.2f}>{limit:.2f}')
            break
        pick=next(order[(turn+i)%len(order)] for i in range(len(order)) if order[(turn+i)%len(order)] in reducible)
        turn=order.index(pick)+1
        caps[pick]-=1;reductions+=1
    report['margin_day_cap']=caps
    report['margin_day_cap_zero']=sorted(s for s,c in caps.items() if c==0)
    report['margin_day_cap_reductions']=reductions
    report['what_if_capped']=rows
    report['capped_qty']={s:max(v['BUY'],v['SELL']) for s,v in qtys.items()}
    report['margin_total_capped']=total
    report['margin_total']=total


def planned_sizes(config,manifest,quote,equity):
    """Contracts size_order would send right now, per market and side: the manifest's prior TR, the current
    mid quote, configured risk bps, fees and exit reserve, clamped to the effective cap. A market the prior-range
    filter does not arm plans 0."""
    out={}
    for m in config.markets:
        item=manifest['markets'][m.name]
        row=dict(prior_tr=item['prior_tr'],skipped=prior_range_skip(config,item))
        if row['skipped']:
            row.update(BUY=0,SELL=0);out[m.execution.symbol]=row;continue
        bid,ask,_=quote(m.name)
        mid=(float(bid)+float(ask))/2
        state=State(manifest['day'],m.name,item['prior_tr'],manifest['score'],half_size=item.get('half_prior_range') is True)
        for side,action in [(1,'BUY'),(-1,'SELL')]:
            plan=size_order(state,m,side,mid,mid,equity,config)
            row[action]=int(plan['qty']);row[f'{action}_risk']=plan['risk']
        row['mid']=mid;out[m.execution.symbol]=row
    return out


def armed_line(config,report,states,prior_tr,score,manifest):
    """ARMED alert: planned contracts at today's ranges (from the 09:25 preflight) and the effective caps."""
    armed=[m for m in config.markets if states[m.name].phase!='SKIPPED']
    skipped={k:s.note for k,s in states.items() if s.phase=='SKIPPED'}
    day_cap=report.get('margin_day_cap')
    if armed:
        planned=report.get('planned_qty') or {}
        sizes=' / '.join(f'{planned.get(m.execution.symbol,"?")} {m.execution.symbol}' for m in armed)
        if day_cap:
            capped=report.get('capped_qty') or {}
            sizes+=', margin-capped to '+' / '.join(str(capped.get(m.execution.symbol,'?')) for m in armed)
        caps='/'.join(str(config.max_contracts_for(m)) for m in armed)
        head=f"planned {sizes} at today's ranges, max {caps}"
    else:
        head='NO MARKET (all skipped; session runs to 16:01 with no orders)'
    money=lambda x:'n/a' if not isinstance(x,(int,float)) else f'${x:,.0f}'
    margin=(f'margin {money(report.get("margin_total"))} of limit {money(report.get("margin_limit"))} '
            f'(uncapped {money(report.get("margin_total_planned"))})' if day_cap else
            f'planned margin {money(report.get("margin_total"))} of limit {money(report.get("margin_limit"))}')
    return (f'ARMED LIVE: {head}; {margin}; '
            f'prior TR {prior_tr}; score {score:.2f}; {range_line(manifest)}'
            +(f'; NOT ARMED {skipped}' if skipped else ''))


class Capture:
    """Hourly compressed append-only captures; bounded open files and disk usage."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.key=None;self.handle=None;self.count=0;self.last={}

    def write(self,event,now):
        key=now.astimezone(NY).strftime('%Y%m%d_%H')
        if key!=self.key:
            self.close();self.key=key
            self.handle=gzip.open(self.root/f'ticks_{key}.jsonl.gz','at',encoding='utf-8')
        event={**event,'received_at':now.isoformat()}
        self.handle.write(json.dumps(event,allow_nan=False)+'\n')
        self.count+=1;self.last[f"{event['market']}:{event['kind']}"]=now.isoformat()

    def flush(self):
        if self.handle:self.handle.flush()

    def close(self):
        if self.handle:self.handle.close();self.handle=None


async def record_other_book(runtime,transport,label):
    """Shadow: informational view of the account's execution contracts; never fails or changes the session."""
    try:
        view=await asyncio.wait_for(book_view(transport),30)
    except Exception as exc:
        view=dict(error=f'{type(exc).__name__}: {exc}')
    runtime.set(f'other_book_{label}',{'at':datetime.now(timezone.utc).isoformat(),**view})


async def run_shadow(config_path,day,risk_path,state_dir,roll_verified=False):
    config=Config.load(config_path)
    clock=lambda:datetime.now(timezone.utc)
    prepare,opening,closing=validate_launch(config,day,clock())
    if not roll_verified:raise ValueError('Reviewed contract roll required')
    root=Path(state_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    runtime=Store(root/'runtime.sqlite',config.fingerprint,day)
    capture=Capture(root/'captures')
    transport=None;service=None;ledger=None
    runtime.set('pid',os.getpid());runtime.set('mode','shadow');runtime.set('session',day)
    runtime.set('python',sys.executable)
    runtime.set('source_hashes',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    runtime.set('config_path',str(Path(config_path).resolve()))
    runtime.set('risk_path',str(Path(risk_path).resolve()))
    runtime.set('settings',{'capital_base':config.shadow_equity,'risk_bps':{m.name:m.risk_bps for m in config.markets},
                         'execution':{m.name:m.execution.symbol for m in config.markets},'port':config.port})
    runtime.set('phase','CONNECTING')
    runtime.event('START',{'pid':os.getpid(),'session':day,'orders_enabled':False})
    print(f'SHADOW starting; orders disabled; session {day}; PID {os.getpid()}',flush=True)
    def feed_error(reason):
        runtime.event('FEED_ERROR',{'reason':str(reason)})
        runtime.set('feed_error',str(reason))
        if service:service.halt(reason)
    def on_capture(event):
        capture.write(event,clock())
        if service and event['kind']=='quote':
            service.broker.update_quote(event['market'],event['bid'],event['ask'],event['time'])
        elif service and event['kind']=='execution_trade':
            service.broker.update_trade(event['market'],event['price'],event['time'])
    def on_tick(name,stamp,price):
        if service:service.tick(name,stamp,price)
    final='STOPPED'
    try:
        while clock()<closing:
            now=clock()
            if (root/'STOP').exists():
                final='STOPPED_BY_REQUEST';break
            if transport is None or not transport.ib.isConnected() or not transport.healthy:
                if service:
                    service.halt('CONNECTION_LOST_AFTER_ARMING');final='HALTED';break
                if transport:transport.close()
                if now>=opening:raise RuntimeError('Opening missed before connection recovered')
                transport=IBKR(config)
                # Hard boundary below the adapter as well: this process cannot transmit.
                def deny_orders(*args,**kwargs):raise PermissionError('Shadow process prohibits broker orders')
                transport.ib.client.placeOrder=deny_orders
                try:
                    await asyncio.wait_for(transport.connect(),30)
                    transport.halt_callback=feed_error
                    transport.subscribe(on_tick,on_capture)
                    await record_other_book(runtime,transport,'connect')
                    runtime.set('phase','WAITING_FOR_PREOPEN')
                    print('Live feed connected; recording while waiting for 09:00 ET preparation',flush=True)
                except Exception as exc:
                    runtime.set('phase','RETRYING_CONNECTION');runtime.set('last_error',str(exc))
                    transport.close();transport=None
                    runtime.set('heartbeat',{'at':clock().isoformat(),'events':capture.count})
                    await asyncio.sleep(10);continue
            if service is None and now>=prepare:
                if now>=opening:raise RuntimeError('Opening missed before inputs prepared')
                runtime.set('phase','PREPARING')
                path=root/'inputs.json'
                if path.exists():
                    # Never replace a retained manifest on restart (and do not re-fetch history).
                    from .inputs import load_manifest
                    manifest=load_manifest(path,config)
                else:
                    bars=await transport.history()
                    range_bars,range_error=await fetch_range_bars(transport)
                    risk=pd.read_parquet(risk_path)
                    manifest=build_manifest(config,day,risk,bars,now=clock(),roll_verified=True,
                                            range_bars=range_bars,range_error=range_error,
                                            prior_day=prior_day_inputs(config,day,root.parent))
                    with path.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
                ledger=Store(root/'trades.sqlite',config.fingerprint,day)
                broker=SimBroker(config,clock)
                service=Service(config,manifest,ledger,broker,clock)
                for name,(bid,ask,stamp) in transport.quotes.items():broker.update_quote(name,bid,ask,stamp)
                runtime.set('inputs',{'hash':manifest['hash'],'score':manifest['score'],
                    'prior_tr':{k:v['prior_tr'] for k,v in manifest['markets'].items()},
                    'prior_range':range_summary(manifest)})
                await record_other_book(runtime,transport,'arm')
                runtime.set('phase','ARMED_SHADOW')
                print(f'SHADOW armed for the cash open; broker order transmission disabled; {range_line(manifest)}',flush=True)
            if service:
                await service.watchdog()
                if service.halted:
                    runtime.set('last_error',ledger.get('halt_reason'));final='HALTED';break
                runtime.set('phase','RUNNING_SHADOW' if now>=opening else 'ARMED_SHADOW')
            capture.flush()
            runtime.set('heartbeat',{'at':clock().isoformat(),'events':capture.count,'last_events':capture.last,
                                    'connected':transport.ib.isConnected(),'healthy':transport.healthy})
            await asyncio.sleep(1)
        else:
            final='SESSION_COMPLETE'
        if service:
            await service.drain()
            if any(s.qty for s in service.states.values()):
                service.halt('RUN_ENDED_WITH_SIMULATED_POSITION');final='HALTED'
    except BaseException as exc:
        final='FAILED';runtime.set('last_error',f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if service:
            await service.cancel_resting_entries()
            await service.drain()
        record_day_r(runtime,ledger,config)
        runtime.set('phase',final);runtime.set('finished_at',clock().isoformat())
        runtime.event('FINISH',{'phase':final,'events':capture.count})
        print(f'SHADOW {final}; captured {capture.count} events',flush=True)
        if transport:
            transport.halt_callback=lambda reason:None
            transport.close()
        capture.close()
        if ledger:ledger.close()
        runtime.close()


ARM_AT=time(9,25)


def prior_live_orders(root,day):
    """Relaunch into a fresh state dir is allowed only when no sibling live journal for the day has orders."""
    import sqlite3
    found=[]
    for d in sorted(root.parent.glob(f'{day}-live*')):
        db=d/'trades.sqlite'
        if d.resolve()==root or not db.exists():continue
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as c:
            if c.execute('SELECT COUNT(*) FROM orders').fetchone()[0]:found.append(d.name)
    return found


async def run_live(config_path,day,risk_path,state_dir,roll_verified=False):
    config=Config.load(config_path)
    clock=lambda:datetime.now(timezone.utc)
    prepare,opening,closing,ack=validate_live_launch(config,day,clock())
    if not roll_verified:raise ValueError('Reviewed contract roll required')
    arm_at=datetime.combine(opening.date(),ARM_AT,NY)
    root=Path(state_dir).resolve()
    blocking=prior_live_orders(root,day)
    if blocking:
        raise RuntimeError(f'Prior live journal(s) for {day} contain orders; reconcile, do not relaunch: {blocking}')
    root.mkdir(parents=True,exist_ok=True)
    runtime=Store(root/'runtime.sqlite',config.fingerprint,day)
    capture=Capture(root/'captures')
    alert=Alerts(f'OpenBreakout LIVE {day}',webhook_url())
    transport=None;gate=None;service=None;ledger=None;manifest=None;owner_server=None
    caps={m.execution.symbol:config.max_contracts_for(m) for m in config.markets}
    runtime.set('pid',os.getpid());runtime.set('mode','live');runtime.set('session',day)
    runtime.set('live_ack',ack);runtime.set('python',sys.executable);runtime.set('alerts',alert.channel)
    runtime.set('source_hashes',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    runtime.set('config_path',str(Path(config_path).resolve()))
    runtime.set('risk_path',str(Path(risk_path).resolve()))
    runtime.set('settings',{'capital_base':config.shadow_equity,'risk_bps':{m.name:m.risk_bps for m in config.markets},
                         'execution':{m.name:m.execution.symbol for m in config.markets},'port':config.port,
                         'client_id':config.client_id,'max_contracts':{m.name:m.max_contracts for m in config.markets},
                         'pilot_max_contracts':config.pilot_max_contracts,'code_cap':LIVE_HARD_MAX_CONTRACTS,
                         'effective_max_contracts':caps})
    runtime.set('phase','CONNECTING')
    runtime.event('START',{'pid':os.getpid(),'session':day,'orders_enabled':'after 09:25 preflight','live_ack':ack})
    print(f'LIVE starting; session {day}; PID {os.getpid()}; max contracts per market {caps}; '
          f'orders blocked until the 09:25 ET preflight passes; alerts {alert.channel}',flush=True)
    def save_report(label,report):
        name=f'preflight_{label}_{clock().astimezone(NY).strftime("%H%M%S")}.json'
        (root/name).write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        runtime.set(f'preflight_{label}',{'ok':report['ok'],'failures':report['failures'],'warnings':report.get('warnings',[]),
                                          'file':name,'other_book':report.get('other_book'),
                                          'what_if_basis':report.get('what_if_basis'),'what_if_qty':report.get('what_if_qty'),
                                          'margin_total':report.get('margin_total'),'margin_limit':report.get('margin_limit'),
                                          'margin_day_cap':report.get('margin_day_cap'),'planned_qty':report.get('planned_qty'),
                                          'capped_qty':report.get('capped_qty'),
                                          'margin_total_planned':report.get('margin_total_planned'),
                                          'margin_total_capped':report.get('margin_total_capped')})
    def feed_error(reason):
        runtime.event('FEED_ERROR',{'reason':str(reason)})
        runtime.set('feed_error',str(reason))
        if service:service.halt(reason)
    def on_capture(event):
        capture.write(event,clock())
    def on_tick(name,stamp,price):
        if service:service.tick(name,stamp,price)
    final='STOPPED';reported=False
    try:
        while clock()<closing:
            now=clock()
            if (root/'STOP').exists():
                final='STOPPED_BY_REQUEST';break
            # After arming, an unhealthy-but-connected feed has already halted new entries through
            # the service callback; keep monitoring protection instead of exiting.
            if transport is None or not transport.ib.isConnected() or (not transport.healthy and not service):
                if service:
                    service.halt('CONNECTION_LOST_AFTER_ARMING');final='HALTED'
                    alert('CONNECTION LOST AFTER ARMING: process stopping; broker-held stop and 15:55 exit remain; CHECK TWS')
                    break
                if transport:transport.close()
                if now>=opening:raise RuntimeError('Opening missed before connection recovered')
                transport=IBKR(config,session=day)
                gate=OrderGate(transport.ib.client)
                try:
                    await asyncio.wait_for(transport.connect(),30)
                    transport.halt_callback=feed_error
                    transport.subscribe(on_tick,on_capture)
                except Exception as exc:
                    runtime.set('phase','RETRYING_CONNECTION');runtime.set('last_error',str(exc))
                    transport.close();transport=None
                    runtime.set('heartbeat',{'at':clock().isoformat(),'events':capture.count})
                    await asyncio.sleep(10);continue
                # manifest is None at the first connect; a reconnect after 09:00 checks the planned sizes.
                report=await preflight(transport,manifest=manifest)
                save_report('connect',report)
                if not report['ok']:
                    runtime.set('last_error',f'PREFLIGHT_FAILED:{report["failures"]}')
                    alert(f'PREFLIGHT FAILED at connect; not arming: {report["failures"]}')
                    final='HALTED_PREFLIGHT';break
                runtime.set('phase','WAITING_FOR_PREOPEN')
                print('Live feed connected and preflight passed; orders blocked until 09:25 ET arming'
                      +(f'; warnings {report["warnings"]}' if report['warnings'] else ''),flush=True)
            if manifest is None and now>=prepare:
                if now>=opening:raise RuntimeError('Opening missed before inputs prepared')
                runtime.set('phase','PREPARING')
                path=root/'inputs.json'
                if path.exists():
                    # Restart: the retained manifest is authoritative; do not spend the pre-open
                    # minutes re-fetching history that would be discarded.
                    from .inputs import load_manifest
                    manifest=load_manifest(path,config)
                else:
                    bars=await transport.history()
                    range_bars,range_error=await fetch_range_bars(transport)
                    risk=pd.read_parquet(risk_path)
                    manifest=build_manifest(config,day,risk,bars,now=clock(),roll_verified=True,
                                            range_bars=range_bars,range_error=range_error,
                                            prior_day=prior_day_inputs(config,day,root.parent))
                    with path.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
                runtime.set('inputs',{'hash':manifest['hash'],'score':manifest['score'],
                    'prior_tr':{k:v['prior_tr'] for k,v in manifest['markets'].items()},
                    'prior_range':range_summary(manifest)})
                runtime.set('phase','PREPARED_WAITING_TO_ARM')
            if manifest is not None and service is None and now>=arm_at:
                if now>=opening:raise RuntimeError('Opening missed before arming')
                report=await preflight(transport,manifest=manifest)
                save_report('arm',report)
                if not report['ok']:
                    runtime.set('last_error',f'PREFLIGHT_FAILED:{report["failures"]}')
                    alert(f'PREFLIGHT FAILED at 09:25; not arming: {report["failures"]}')
                    final='HALTED_PREFLIGHT';break
                ledger=Store(root/'trades.sqlite',config.fingerprint,day)
                ledger.set('live_ack',ack)
                # Margin day cap (per session, recomputed at every arming): market name -> whole contracts.
                day_cap=report.get('margin_day_cap')
                name_caps={m.name:day_cap.get(m.execution.symbol) for m in config.markets} if day_cap else None
                runtime.set('margin_day_cap',name_caps)
                detail={k:report.get(k) for k in ('planned_qty','capped_qty','margin_total_planned','margin_total_capped',
                                                  'margin_limit','margin_day_cap_reductions')} if day_cap else None
                service=Service(config,manifest,ledger,transport,clock,margin_day_cap=name_caps,margin_detail=detail)
                service.notify=alert
                await service.watchdog()
                if service.halted:
                    runtime.set('last_error',ledger.get('halt_reason'));final='HALTED';break
                gate.open=True
                from broker_runtime.owner_connection import OwnerServer
                owner_server=OwnerServer(transport, service.operator_order)
                await owner_server.start()
                runtime.set('phase','ARMED_LIVE')
                alert(armed_line(config,report,service.states,runtime.get('inputs')['prior_tr'],manifest['score'],manifest))
            if service:
                await service.watchdog()
                if service.halted and not reported:
                    reported=True;runtime.set('last_error',ledger.get('halt_reason'))
                    print('HALTED: no new entries; broker-held protection retained; inspect journal and TWS',flush=True)
                runtime.set('phase','HALTED_MONITORING' if service.halted else ('RUNNING_LIVE' if now>=opening else 'ARMED_LIVE'))
            capture.flush()
            runtime.set('heartbeat',{'at':clock().isoformat(),'events':capture.count,'last_events':capture.last,
                                    'connected':transport.ib.isConnected(),'healthy':transport.healthy,
                                    'orders_open':bool(gate and gate.open)})
            await asyncio.sleep(1)
        else:
            final='SESSION_COMPLETE'
        if service:
            await service.drain()
            if any(s.qty for s in service.states.values() if s.market not in service.manual_markets):
                service.halt('RUN_ENDED_WITH_LIVE_POSITION');final='HALTED'
                alert('RUN ENDED WITH A LIVE POSITION IN THE JOURNAL: FLATTEN/INSPECT IN TWS NOW')
            elif service.halted and final=='SESSION_COMPLETE':
                final='SESSION_COMPLETE_HALTED'
    except BaseException as exc:
        final='FAILED';runtime.set('last_error',f'{type(exc).__name__}: {exc}')
        alert(f'LIVE PROCESS FAILED: {type(exc).__name__}: {exc}')
        raise
    finally:
        if service:
            await service.cancel_resting_entries()
            await service.drain()
        if owner_server:await owner_server.close()
        if gate:gate.open=False
        record_day_r(runtime,ledger,config)
        runtime.set('phase',final);runtime.set('finished_at',clock().isoformat())
        runtime.event('FINISH',{'phase':final,'events':capture.count})
        print(f'LIVE {final}; captured {capture.count} events',flush=True)
        if final!='SESSION_COMPLETE':
            alert(f'LIVE PROCESS EXITING: {final}')
        if transport:
            transport.halt_callback=lambda reason:None
            transport.close()
        capture.close()
        if ledger:ledger.close()
        runtime.close()
        # Exit paths alert immediately before returning; let those posts finish.
        alert.flush(5.)
