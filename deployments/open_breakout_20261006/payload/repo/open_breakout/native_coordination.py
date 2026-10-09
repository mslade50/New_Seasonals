"""Concrete owner adapter and async driver. Importing never connects.

Reads use the already-owned IB-style connection; sends use the existing owner
transport. Native automatic closing is deliberately unavailable: no verified
protection-preserving order-type change is established for the current broker.
An explicit offline fixture qualification exercises the complete candidate
same-ID close and recovery implementation without any broker connection.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, time
import json
import math
from types import SimpleNamespace

from intraday_coordination import CoordinationError, FAMILY, NORMALIZED_STATUSES, OrderView, Snapshot, stamp


def whole(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
        raise CoordinationError('Unknown/non-whole native quantity or identity')
    value=int(value)
    if value < (1 if positive else 0):
        raise CoordinationError('Negative/zero native quantity or identity')
    return value


def wire(trade,*,allow_zero_order_id=False):
    o=trade.order
    return dict(account=str(o.account), con_id=whole(trade.contract.conId,positive=True),
                client_id=whole(o.clientId), order_id=whole(o.orderId,positive=not allow_zero_order_id), perm_id=whole(o.permId,positive=True))


@dataclass(frozen=True)
class OfflineCloseQualification:
    """A mock assertion, explicitly NOT proof about TWS/IBKR behavior."""
    same_id_never_reopens_terminal: bool
    old_stop_retained_until_acceptance: bool
    rejection_keeps_old_stop: bool
    unchanged_total_and_oca_serialize_fills: bool


class NativeOwnerAdapter:
    def __init__(self,service,*,qualification=None,timeout=2.,filter_factory=None):
        self.service=service;self.transport=service.broker;self.ledger=service.coordination
        self.qualification=qualification;self.timeout=timeout;self.cache=None;self.raw={};self.reason=''
        self.filter_factory=filter_factory or self._filter
        self.generation=None

    @staticmethod
    def _filter(account):
        from ib_insync import ExecutionFilter  # a data object; never constructs/connects an IB client
        return ExecutionFilter(acctCode=account)

    def qualified(self):
        q=self.qualification
        return bool(getattr(self.transport,'offline_fixture',False) is True and q
                    and all(getattr(q,k) is True for k in q.__dataclass_fields__))

    def generation_now(self):
        return getattr(self.transport,'_exec_generation',None)

    def _current(self,scope):
        if (not self.cache or self.cache.scope != scope or self.generation != self.generation_now()
                or scope.family in self.service.manual_markets
                or not 0 <= (stamp(self.service.clock())-stamp(self.cache.at)).total_seconds() <= 3):
            raise CoordinationError('Owner evidence changed/staled since asynchronous snapshot')
        return self.cache

    async def refresh(self,scope):
        """Fresh, sequential broker reads under the transport's existing async lock.

        The coordination file lock is NEVER held over these awaits. Closed order
        receipts must be returned, not inferred from absence in open orders.
        Position, executions, owner journal and status counters must reconcile.
        """
        self.cache=None;self.raw={};self.reason=''
        try:
            async def read():
                async with self.transport.snapshot_lock:
                    ib=self.transport.ib
                    if ib.client.clientId!=self.service.config.client_id or not ib.isConnected():
                        raise CoordinationError('Owner connection is disconnected or its client identity differs')
                    opened=list(await ib.reqAllOpenOrdersAsync())
                    completed=list(await ib.reqCompletedOrdersAsync(apiOnly=False))
                    fills=list(await ib.reqExecutionsAsync(self.filter_factory(scope.account)))
                    positions=list(await ib.reqPositionsAsync())
                return opened,completed,fills,positions
            for _ in range(3):
                before_generation=self.generation_now()
                opened,completed,fills,positions=await asyncio.wait_for(read(),self.timeout)
                before_orders=set(self.service.store.orders())
                try:snap=self._normalize(scope,opened,completed,fills,positions)
                except CoordinationError:
                    if before_generation!=self.generation_now():
                        await asyncio.sleep(0)
                        continue
                    raise
                if before_generation!=self.generation_now() or before_orders!=set(self.service.store.orders()):
                    await asyncio.sleep(0)
                    continue
                self.cache=snap;self.generation=self.generation_now()
                return snap
            raise CoordinationError('Owner executions/orders kept changing across snapshot; no stable evidence')
        except Exception as exc:
            self.reason=str(exc) or type(exc).__name__
            m=self.service.markets[scope.family]
            self.cache=Snapshot(scope,self.service.clock(),m.execution.con_id,0,0,(),False,False,
                                scope.family in self.service.manual_markets,False)
            self.generation=self.generation_now()
            return self.cache

    def _normalize(self,scope,opened,completed,fills,positions):
        if self.service.config.account != scope.account or self.service.states[scope.family].day != scope.day:
            raise CoordinationError('Owner account/session differs from coordination scope')
        market=self.service.markets[scope.family];cid=market.execution.con_id
        # Corrections replace their original execution stem; conflicting duplicate
        # payloads do not silently become another fill.
        latest={}
        for fill in fills:
            e=fill.execution
            if str(e.acctNumber)!=scope.account:continue
            if FAMILY.get(fill.contract.symbol)!=scope.family:continue
            if stamp(e.time).date().isoformat()!=scope.day:continue
            if e.side not in {'BOT','SLD'}:raise CoordinationError('Unknown execution side')
            key=str(e.execId).rpartition('.')[0] or str(e.execId)
            suffix=str(e.execId).rpartition('.')[2]
            p=str(e.orderRef or '').split('|')
            owned_execution=len(p)>=4 and p[2]=='OpenBreakout' and p[3]==scope.day
            payload=(whole(e.shares,positive=True),whole(e.permId,positive=True),whole(e.orderId,positive=owned_execution),
                     whole(e.clientId),str(e.orderRef),whole(fill.contract.conId,positive=True),e.side)
            if key in latest and suffix==latest[key][0] and payload!=latest[key][2]:
                raise CoordinationError('Conflicting duplicate execution receipt')
            if key not in latest or suffix>latest[key][0]:latest[key]=(suffix,fill,payload)
        executions=[row[1] for row in latest.values()]
        journal_ids=set(self.service.store.fill_ids())
        for key,(suffix,fill,payload) in latest.items():
            if any((old.rpartition('.')[0] or old)==key and old!=str(fill.execution.execId) for old in journal_ids):
                raise CoordinationError('Execution correction requires owner journal reconciliation, never another additive fill')
        stored=self.service.store.orders();records={};sources={}
        identity_records=self.service.store.get('coord_native_identities',{})
        for source,rows in (('open',opened),('completed',completed)):
            for trade in rows:
                if str(trade.order.account)!=scope.account:continue
                if FAMILY.get(trade.contract.symbol)!=scope.family:continue
                p=str(trade.order.orderRef or '').split('|')
                is_owned=len(p)>=4 and p[2]=='OpenBreakout' and p[3]==scope.day
                if source=='completed':
                    trade=copy.deepcopy(trade)
                    # Installed ib_insync.completedOrder constructs an empty
                    # OrderStatus and leaves clientId/orderId at default zero.
                    # Native completed fields provide permId and filledQuantity.
                    # Rejoin ONLY exact independently proved identities, never
                    # infer an API ID from an orderRef/absence/zero default.
                    candidates=[]
                    for cached in self.transport.trades.values():
                        try:identity=wire(cached)
                        except CoordinationError:continue
                        if (identity['account']==trade.order.account and identity['con_id']==trade.contract.conId
                                and identity['perm_id']==trade.order.permId and cached.order.orderRef==trade.order.orderRef):
                            candidates.append(identity)
                    for saved in identity_records.values():
                        if (saved['account']==trade.order.account and saved['con_id']==trade.contract.conId
                                and saved['perm_id']==trade.order.permId and saved['ref']==trade.order.orderRef):
                            candidates.append({k:saved[k] for k in ('account','con_id','client_id','order_id','perm_id')})
                    for f in executions:
                        e=f.execution
                        if (e.acctNumber==trade.order.account and f.contract.conId==trade.contract.conId
                                and e.permId==trade.order.permId and e.orderRef==trade.order.orderRef):
                            candidates.append(dict(account=e.acctNumber,con_id=f.contract.conId,
                                client_id=whole(e.clientId),order_id=whole(e.orderId,positive=is_owned),perm_id=whole(e.permId,positive=True)))
                    unique={tuple(row.items()):row for row in candidates}
                    if len(unique)>1:raise CoordinationError('Completed permanent ID maps to conflicting API owner identities')
                    if unique:
                        identity=next(iter(unique.values()))
                        if (trade.order.orderId not in {0,identity['order_id']}
                                or trade.order.clientId not in {0,identity['client_id']}):
                            raise CoordinationError('Completed IDs differ from proved native identity')
                        trade.order.orderId=identity['order_id'];trade.order.clientId=identity['client_id']
                    elif is_owned:
                        raise CoordinationError('Completed owner receipt lacks proved client/API identity; no cancellation inference')
                    filled=whole(getattr(trade.order,'filledQuantity',float('nan')))
                    if trade.orderStatus.filled not in {0,filled}:
                        raise CoordinationError('Completed fillQuantity differs from populated status counter')
                    trade.orderStatus.filled=filled
                identity=wire(trade,allow_zero_order_id=not is_owned);key=tuple(identity.values())
                if key in records:
                    previous=records[key]
                    if (previous.orderStatus.status != trade.orderStatus.status
                            or previous.orderStatus.filled != trade.orderStatus.filled
                            or previous.order.totalQuantity != trade.order.totalQuantity):
                        raise CoordinationError('Open/completed receipts disagree; wait for stable evidence')
                records[key]=trade;sources[key]=source
                if is_owned:
                    ref=str(trade.order.orderRef)
                    if (identity['client_id']!=self.service.config.client_id or identity['order_id'] not in stored
                            or stored[identity['order_id']]['body']['ref']!=ref):
                        raise CoordinationError('Completed/open identity lacks matching owner journal')
                    saved={**identity,'ref':ref}
                    identity_key=scope.key+f':{identity["client_id"]}:{identity["order_id"]}'
                    if identity_key in identity_records and identity_records[identity_key]!=saved:
                        raise CoordinationError('Persisted native permanent identity changed')
                    identity_records[identity_key]=saved
        self.service.store.set('coord_native_identities',identity_records)
        views=[];owned=0;seen_exec=set()
        for trade in records.values():
            o=trade.order
            parts=str(o.orderRef or '').split('|');strategy=parts[2] if len(parts)>=4 else 'Manual'
            day=parts[3] if len(parts)>=4 else scope.day
            owned_order=(strategy=='OpenBreakout' and day==scope.day and FAMILY.get(trade.contract.symbol)==scope.family)
            identity=wire(trade,allow_zero_order_id=not owned_order);oid=identity['order_id'];key=tuple(identity.values())
            row=stored.get(oid) if owned_order else None
            if owned_order and (not row or row['market']!=scope.family or str(o.orderRef)!=row['body']['ref']
                    or identity['client_id']!=self.service.config.client_id or identity['con_id']!=cid
                    or row['body'].get('account')!=scope.account):
                raise CoordinationError('Native breakout receipt lacks exact owner journal provenance')
            matches=[]
            for f in executions:
                e=f.execution
                if (e.acctNumber==identity['account'] and int(e.permId)==identity['perm_id']):
                    if (int(e.orderId)!=oid or int(e.clientId)!=identity['client_id']
                            or f.contract.conId!=identity['con_id'] or e.orderRef!=o.orderRef
                            or ('BUY' if e.side=='BOT' else 'SELL')!=o.action):
                        raise CoordinationError('Execution identity differs from order receipt')
                    matches.append(f)
            executed=sum(whole(f.execution.shares,positive=True) for f in matches)
            status=str(trade.orderStatus.status)
            reported=whole(trade.orderStatus.filled)
            # An OCA proportional reduction may produce a zero-size cancelled
            # sibling. Only an explicit completed cancellation permits zero;
            # it is never accepted as an active order or a zero-size fill.
            total=whole(o.totalQuantity,positive=status not in {'Cancelled','ApiCancelled'})
            if executed!=reported or executed>total:
                raise CoordinationError('Order status fill counters differ from complete execution receipts')
            if status in {'Inactive','REJECTED'}:
                # Explicit system rejection + completed receipt + no active row;
                # never normalize merely cached Inactive into a terminal cancel.
                codes={getattr(log,'errorCode',None) for log in getattr(trade,'log',[])}
                cached=self.transport.trades.get(oid)
                if cached is not None and wire(cached)==identity:
                    codes.update(getattr(log,'errorCode',None) for log in getattr(cached,'log',[]))
                if sources[key]!='completed' or not codes.intersection({201,203}):
                    raise CoordinationError('Inactive is not a confirmed rejection/cancellation')
                status='Rejected'
            if status not in NORMALIZED_STATUSES:
                raise CoordinationError('Unknown native order status')
            terminal=status in {'Filled','Cancelled','ApiCancelled','Rejected'}
            if terminal and sources[key]!='completed':
                raise CoordinationError('Terminal status lacks a completed-order receipt')
            if status=='Filled' and executed!=total:
                raise CoordinationError('Filled status without full actual executions')
            owner_side=int(row['body']['side']) if row and row['role']=='ENTRY' else (
                1 if len(parts)>=2 and parts[1]=='BUY' else -1 if len(parts)>=2 and parts[1]=='SELL' else
                1 if o.action=='BUY' else -1)
            if row and row['role']!='ENTRY' and (1 if o.action=='BUY' else -1)!=-owner_side:
                raise CoordinationError('Owned exit direction differs from its entry provenance')
            role=row['role'] if row else 'OTHER'
            state=self.service.states[scope.family]
            if (row and role=='STOP' and oid==state.stop_order and status not in {'Filled','Cancelled','ApiCancelled','Rejected'}
                    and o.orderType=='STP' and getattr(o,'auxPrice',None)!=state.stop):
                raise CoordinationError('Current protective stop price differs from owner state; manual/unknown modification')
            token=f'OB:{identity["client_id"]}:{oid}'
            terms=dict(kind=o.orderType,total_qty=total,tif=o.tif,ref=o.orderRef,stop=getattr(o,'auxPrice',0),
                       good_after=getattr(o,'goodAfterTime',''),oca_type=getattr(o,'ocaType',0),
                       account=o.account,side=1 if o.action=='BUY' else -1)
            view=OrderView(token,scope.account,day,trade.contract.symbol,identity['con_id'],identity['client_id'],
                           oid,identity['perm_id'],strategy,role,owner_side,status,0 if terminal else total-executed,
                           executed,executed,str(getattr(o,'ocaGroup','') or ''),terms)
            views.append(view);self.raw[token]=trade
            if owned_order:
                owned+=sum((1 if f.execution.side=='BOT' else -1)*whole(f.execution.shares,positive=True) for f in matches)
                seen_exec.update(str(f.execution.execId) for f in matches)
                if role=='ENTRY':
                    durable=self.ledger.read(scope)['entries'].get(token)
                    if not durable:raise CoordinationError('Owner entry receipt lacks durable arbitration send intent')
                    if durable['identity']!=view.identity:self.ledger.bind_entry_identity(scope,token,view.identity)
        # Every journalled order needs an exact fresh open or completed receipt.
        suppressed={json.loads(row[0])['id'] for row in self.service.store.db.execute(
            "SELECT body FROM events WHERE kind='COORD_ENTRY_SUPPRESSED'")}
        durable_entries=self.ledger.read(scope)['entries']
        for oid,row in stored.items():
            if row['market']!=scope.family or row['body'].get('account')!=scope.account:continue
            if not any(v.order_id==oid and v.client_id==self.service.config.client_id for v in views):
                token=f'OB:{self.service.config.client_id}:{oid}'
                if row['role']=='ENTRY' and oid in suppressed and token not in durable_entries:
                    # This exact entry was refused BEFORE its durable send
                    # intent/transport call. This is local no-send evidence,
                    # not an inferred terminal broker cancellation.
                    continue
                raise CoordinationError('Journal order absent from broker open/completed receipts')
        for f in executions:
            e=f.execution;p=str(e.orderRef or '').split('|')
            if (len(p)>=4 and p[2]=='OpenBreakout' and p[3]==scope.day
                    and FAMILY.get(f.contract.symbol)==scope.family and str(e.execId) not in seen_exec):
                raise CoordinationError('Owned execution has no exact terminal/open order receipt')
        # Deliver queried owner executions through the ORIGINAL idempotent fill
        # handler; never invent a state delta from an aggregate position.
        for f in executions:
            e=f.execution
            if (str(e.execId) in seen_exec and int(e.clientId)==self.service.config.client_id
                    and str(e.execId) not in self.service.store.fill_ids()):
                self.transport._fill(None,f)
        state=self.service.states[scope.family]
        if state.side*state.qty != owned:
            raise CoordinationError('Native attributed executions differ from owner journal quantity')
        position_rows=[p for p in positions if str(p.account)==scope.account and p.contract.conId==cid]
        if len(position_rows)>1:raise CoordinationError('Duplicate native account position')
        account_qty=int(position_rows[0].position) if position_rows else 0
        if position_rows and position_rows[0].position!=account_qty:raise CoordinationError('Non-whole account net')
        return Snapshot(scope,self.service.clock(),cid,owned,account_qty,tuple(views),True,True,
                        scope.family in self.service.manual_markets,self.qualified())

    def exact(self,scope,view):
        self._current(scope)
        native=self.transport.trades.get(view.order_id)
        if native is None:
            # Cold owner cache: only hydrate from a fresh exact receipt read on
            # the SAME configured client. Never bind/edit another client's order.
            native=self.raw.get(view.token)
            if native is not None and view.client_id==self.service.config.client_id:
                self.transport.trades[view.order_id]=native
        if not native or wire(native)!={k:view.identity[k] for k in ('account','con_id','client_id','order_id','perm_id')}:
            raise CoordinationError('Current native owner order identity changed')
        if str(native.orderStatus.status) in {'Filled','Cancelled','ApiCancelled','Inactive','REJECTED'}:
            raise CoordinationError('Current native order is terminal/unknown; refresh before mutation')
        if whole(native.orderStatus.filled)!=view.execution_filled:
            raise CoordinationError('An execution arrived since snapshot; refresh, do not resize from stale quantity')
        o=native.order
        if (o.orderRef!=view.terms['ref'] or o.orderType!=view.terms['kind']
                or o.totalQuantity!=view.terms['total_qty'] or o.ocaGroup!=view.oca or o.ocaType!=view.terms['oca_type']
                or o.tif!=view.terms['tif'] or getattr(o,'auxPrice',0)!=view.terms['stop']
                or getattr(o,'goodAfterTime','')!=view.terms['good_after']):
            raise CoordinationError('Native terms changed since snapshot; no mutation from stale order echo')
        return native

    def cancel_exact(self,scope,view):
        self.exact(scope,view)
        state=self.service.states[scope.family]
        if state.qty and view.role!='ENTRY':
            raise CoordinationError('This adapter never cancels protection for a filled position')
        # Expected OCA sibling cancellations are tracked by the owner.
        for v in self.cache.orders:
            if v.strategy=='OpenBreakout' and v.oca==view.oca and v.role==view.role:
                self.service.cancel_requested.add(v.order_id)
        self.service.cancel_requested.add(view.order_id)
        self.transport.cancel(view.order_id)

    def modify_stop_to_market(self,scope,stop):
        if not self.qualified():
            raise CoordinationError('No qualified native protection-preserving close; retain existing exits')
        native=self.exact(scope,stop)
        if native.order.orderType!='STP':raise CoordinationError('Selected original protection is not STP')
        for view in self.cache.orders:
            if (view.strategy=='OpenBreakout' and view.role=='TIME' and view.active
                    and view.con_id==stop.con_id and view.owner_side==stop.owner_side):
                self.exact(scope,view)
        body=copy.deepcopy(self.service.store.orders()[stop.order_id]['body'])
        # Keep the ORIGINAL native total (including already filled shares), ID,
        # orderRef and OCA membership. Never increase/resubmit a residual size.
        body.update(kind='MKT',qty=whole(native.order.totalQuantity,positive=True))
        body.pop('stop',None)
        self.service.cancel_requested.add(stop.order_id)  # a reject must not trigger an independent emergency close
        self.service.store.event('COORD_MODIFY_EXISTING_STOP_INTENT',dict(id=stop.order_id,identity=stop.identity,body=body))
        self.transport.send(stop.order_id,self.service.markets[scope.family],body)

    def restore_stop(self,scope,quantity,original,operation_id):
        if not self.qualified():raise CoordinationError('Residual restoration is offline-qualified only')
        self._current(scope)
        state=self.service.states[scope.family];market=self.service.markets[scope.family]
        body=copy.deepcopy(original)
        body.update(kind='STP',qty=quantity)
        oid=self.transport.next_id()
        # Durable reservation is external to broker send and source state points
        # to this exact new protection ID before any callback can arrive.
        self.service.store.order(oid,scope.family,'STOP',body)
        self.service.store.event('COORD_RESTORE_STOP_INTENT',dict(operation_id=operation_id,id=oid,body=body))
        state.stop_order=oid;self.service.store.save(state)
        self.transport.send(oid,market,body)
        return oid


class AsyncOwnerCoordinator:
    """Refresh asynchronously, then do one short locked state/send transition.

    Unlike the legacy cancel-then-MKT model, this never removes a live protective
    exit to prepare a new close. Its native default is manual reconciliation.
    The fixture-qualified alternative modifies the existing STOP's same native
    identity; it does not append a market order to an already-active OCA group.
    """
    def __init__(self,adapter,*,close_timeout=3.):
        self.adapter=adapter;self.ledger=adapter.ledger;self.lock=asyncio.Lock();self.close_timeout=close_timeout

    async def poll(self,scope,now):
        async with self.lock:
            await self.adapter.refresh(scope)
            with self.ledger.locked():
                body=self.ledger._read(scope);op=body['operation'];signal=body['signal']
                if not op:return 'no_signal'
                if op['phase']=='reconcile':return 'reconcile'
                def save(phase=None,reason=None):
                    if phase:op['phase']=phase
                    if reason:op['reason']=reason
                    if op['phase']=='done':op['settled_at']=stamp(now).isoformat()
                    self.ledger._save(scope,body);return op['phase']
                def fail(reason):return save('reconcile',reason)
                snap=self.adapter.cache
                close=op.get('protected_close')
                protection_followup=bool(close and close.get('submitted') and op['phase'] in {
                    'protected_working','protected_restore','protected_restore_pending','cleanup'})
                if self.adapter.service.halted and not protection_followup:
                    return fail('Owner hard halt; no automatic coordination mutations; existing protection retained')
                if (signal['status']!='eligible' or stamp(now).date().isoformat()!=scope.day
                        or not snap.complete or not snap.ownership_known or snap.manual_control):
                    return fail('Owner evidence/signal/manual state is unsafe: '+self.adapter.reason)
                try:self.adapter._current(scope)
                except Exception as exc:return fail(str(exc))
                opposite=-signal['side'];qty=abs(snap.owned_qty) if snap.owned_qty*opposite>0 else 0
                own=[v for v in snap.orders if v.strategy=='OpenBreakout' and v.day==scope.day
                     and v.account==scope.account and FAMILY.get(v.symbol)==scope.family and v.owner_side==opposite]
                entries=[v for v in own if v.role=='ENTRY' and v.active]
                # Admission was fenced at publication. Cancellation only after
                # exact fresh owner evidence, with at-most-once durable intent.
                for view in entries:
                    durable=body['entries'].get(view.token)
                    if not durable or durable['identity']!=view.identity:return fail('Entry owner identity differs')
                    if view.token not in op['cancelled']:
                        op['cancelled'][view.token]='intent';save()
                        try:self.adapter.cancel_exact(scope,view)
                        except Exception as exc:return fail('Entry cancel delivery/state unknown: '+str(exc))
                if entries:
                    requested=op.setdefault('cancel_started',stamp(now).isoformat())
                    if (stamp(now)-stamp(datetime.fromisoformat(requested))).total_seconds()>self.close_timeout:
                        return fail('Entry cancellation timeout; terminal receipts required; protection retained')
                    return save('cancel_entries')
                exits=[v for v in own if v.role in {'STOP','TIME'} and v.active]
                if op['phase']=='done':
                    if qty or exits:return fail('Late opposing owned execution/order after settlement')
                    return save('done')
                if op['phase']=='cancel_entries':return save('protected_prepare' if qty else 'cleanup')
                if op['phase']=='protected_prepare':
                    if not qty:return save('cleanup')
                    if (not snap.close_capable or snap.account_qty*opposite<qty
                            or any(v.active and v.con_id==snap.con_id and v not in own for v in snap.orders)):
                        return fail('No qualified protected close, competing order or account net ambiguity; existing exits retained')
                    if stamp(now).time().replace(tzinfo=None)>time(9,31,20):
                        return fail('No protected close begun after original Legend deadline; existing exits retained')
                    stops=[v for v in exits if v.role=='STOP' and v.remaining==qty and v.terms['kind']=='STP']
                    timed=[v for v in exits if v.role=='TIME' and v.remaining==qty]
                    if (len(stops)!=1 or len(timed)!=1 or not stops[0].oca or stops[0].oca!=timed[0].oca
                            or stops[0].terms['oca_type']!=2 or timed[0].terms['oca_type']!=2
                            or timed[0].terms['good_after']!=scope.day.replace('-','')+' 15:55:00 America/New_York'):
                        return fail('Exact original STOP/TIME OCA-2 topology/coverage is not verified')
                    stop=stops[0]
                    op['protected_close']=dict(identity=stop.identity,token=stop.token,quantity=qty,
                        before_filled=stop.execution_filled,started=stamp(now).isoformat(),submitted=False,
                        original_body=copy.deepcopy(self.adapter.service.store.orders()[stop.order_id]['body']))
                    return save('protected_submit')
                close=op.get('protected_close')
                if op['phase']=='protected_submit':
                    stop=next((v for v in own if close and v.token==close['token']),None)
                    if (not stop or stop.identity!=close['identity'] or not stop.active
                            or stop.execution_filled!=close['before_filled'] or qty!=close['quantity']):
                        return fail('Original stop filled/changed before close modification; refresh manually, no new close')
                    if (snap.account_qty*opposite<qty
                            or any(v.active and v.con_id==snap.con_id and v not in own for v in snap.orders)):
                        return fail('Competing order/account net changed before close; existing stop retained')
                    timed=[v for v in exits if v.role=='TIME' and v.remaining==qty]
                    if (len(timed)!=1 or stop.terms['kind']!='STP' or stop.oca!=timed[0].oca
                            or stop.terms['oca_type']!=2 or timed[0].terms['oca_type']!=2
                            or timed[0].terms['good_after']!=scope.day.replace('-','')+' 15:55:00 America/New_York'):
                        return fail('Protection topology changed before modification; no close')
                    close['submitted']=True;save('protected_working')
                    try:self.adapter.modify_stop_to_market(scope,stop)
                    except Exception as exc:return fail('Same-ID modification delivery unknown; do not repeat: '+str(exc))
                    return op['phase']
                if op['phase']=='protected_working':
                    stop=next((v for v in own if close and v.token==close['token']),None)
                    if not stop or stop.identity!=close['identity']:return fail('Exact close/stop receipt missing or identity changed')
                    delta=stop.execution_filled-close['before_filled']
                    if delta<0 or delta>close['quantity'] or qty!=close['quantity']-delta:
                        return fail('Close actual fills and attributed residual do not reconcile')
                    if not stop.active:
                        return save('protected_restore' if qty else 'cleanup')
                    if stop.terms['kind']=='STP':
                        # A definitively refused modification may retain the old
                        # stop. Never submit another close or assume it succeeded.
                        return fail('Modification not accepted; original protective stop still working; manual review')
                    if stop.terms['kind']!='MKT':return fail('Unexpected native close order type')
                    if (stamp(now)-stamp(datetime.fromisoformat(close['started']))).total_seconds()>self.close_timeout:
                        return fail('Working/partial close timeout; no competing resend/restoration; immediate manual review')
                    return save()
                if op['phase']=='protected_restore':
                    if not qty:return save('cleanup')
                    if (snap.account_qty*opposite<qty or any(v.active and v.role=='STOP' for v in own)
                            or any(v.active and v.con_id==snap.con_id and v not in own for v in snap.orders)):
                        return fail('Residual restoration is ambiguous; no new protection inferred')
                    # Keep any known original time sibling. Reject a mismatched
                    # outstanding quantity rather than recreating/resizing it.
                    if any(v.role=='TIME' and v.remaining!=qty for v in exits):return fail('Timed sibling residual differs')
                    save('protected_restore_pending')
                    try:close['restored_order_id']=self.adapter.restore_stop(scope,qty,close['original_body'],op['id']+':restore')
                    except Exception as exc:return fail('Restoration delivery unknown; never repeat: '+str(exc))
                    return save()
                if op['phase']=='protected_restore_pending':
                    restored=next((v for v in exits if v.order_id==close.get('restored_order_id')),None)
                    if not restored or restored.role!='STOP' or restored.remaining!=qty or restored.terms['kind']!='STP':
                        return fail('Restored protective stop not acknowledged; immediate manual review')
                    return fail('Terminal partial/rejected close: verified residual stop restored once; manual review')
                if op['phase']=='cleanup':
                    if qty:return fail('Late opposing exposure during exit cleanup')
                    for view in exits:
                        if view.token not in op['cancelled']:
                            op['cancelled'][view.token]='intent';save()
                            try:self.adapter.cancel_exact(scope,view)
                            except Exception as exc:return fail('Owned orphan exit cancellation unknown: '+str(exc))
                    if exits:return save()
                    return save('done')
                return fail('Legacy/unsupported close phase cannot remove protection')

    async def recover(self,scope,now):
        """Read-only recovery of one uncertain submitted same-ID close.

        Only a terminal exact order plus actual execution/journal agreement may
        resume cleanup or one residual restoration; never repeats modification.
        """
        async with self.lock:
            snap=await self.adapter.refresh(scope)
            with self.ledger.locked():
                body=self.ledger._read(scope);op=body['operation'];close=(op or {}).get('protected_close')
                if not close or not close.get('submitted') or op['phase']!='reconcile':return 'reconcile'
                if not snap.complete or not snap.ownership_known or snap.manual_control:return 'reconcile'
                view=next((v for v in snap.orders if v.token==close['token']),None)
                if not view or view.identity!=close['identity'] or view.active:return 'reconcile'
                opposite=-body['signal']['side'];qty=abs(snap.owned_qty) if snap.owned_qty*opposite>0 else 0
                if qty!=close['quantity']-(view.execution_filled-close['before_filled']):return 'reconcile'
                if qty and snap.account_qty*opposite<qty:return 'reconcile'
                if body['signal']['status']!='eligible':return 'reconcile'
                op['phase']='protected_restore' if qty else 'cleanup'
                op['reason']='Exact late terminal receipt recovered; no repeated close modification'
                self.ledger._save(scope,body);return op['phase']
