"""Opt-in durable lifecycle. SDK status alone is never broker acceptance evidence.

All wire actions remain in the existing owner. This module only records/fences
intents and consumes explicit native response evidence. A restart never resends.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, time, timezone
import hashlib
import json
import math

from .strategy import NY, snap, size_order
from .order_policy import validate_policy

ACCEPTED = {'Submitted', 'PreSubmitted'}
TERMINAL = {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}
VERSION = 'open-breakout-reliability-20261007-v2'



def body_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Lifecycle:
    """One immutable send intent and one cancel intent per exact owner order ID.

    Persisted send_started/cancel_started are write-ahead uncertainty fences, not
    proof that bytes reached the broker. Neither fence may be retried on restart.
    Native terminal callbacks need a later complete execution cycle to close.
    """
    KEY = 'order_lifecycle_v2'

    def __init__(self, store, account, client, session, clock):
        self.store, self.account, self.client, self.session, self.clock = store, account, client, session, clock
        self.revision = int(store.get('order_lifecycle_revision', 0))

    def rows(self):
        return self.store.get(self.KEY, {})

    def _save(self, row, event):
        rows = self.rows(); rows[str(row['id'])] = row
        self.revision += 1
        self.store.set(self.KEY, rows)
        self.store.set('order_lifecycle_revision', self.revision)
        self.store.event(event, row)
        return deepcopy(row)

    def intent(self, oid, market, role, con_id, body, risk=0):
        if str(oid) in self.rows():
            raise ValueError('Order ID already has a durable intent; resend forbidden')
        row = dict(id=oid, account=self.account, client_id=self.client, session=self.session,
                   market=market, role=role, con_id=con_id, ref=body['ref'], body=deepcopy(body),
                   body_sha256=body_hash(body), revision=1, risk=risk, phase='INTENT',
                   created_at=self.clock().isoformat(), send_started=False, cancel_started=False,
                   perm_id=0, native=None, echo=None, terminal=None, closed=False, fills={},
                   filled=0, uncertain=False)
        return self._save(row, 'DURABLE_ORDER_INTENT')

    def send_started(self, oid):
        row = self.rows()[str(oid)]
        if row['send_started'] or row['cancel_started']:
            raise ValueError('Durable send fence already consumed')
        row.update(send_started=True, phase='AWAITING_NATIVE', uncertain=True)
        return self._save(row, 'SEND_FENCE')

    def modify_started(self, oid, body):
        row = self.rows()[str(oid)]
        if row['closed'] or row['cancel_started']:
            raise ValueError('Cannot modify a closed/cancelled order')
        # An unconfirmed revision cannot be replayed after a restart.
        row.update(body=deepcopy(body), body_sha256=body_hash(body), revision=row['revision']+1,
                   echo=None, native=None, phase='AWAITING_NATIVE_REVISION', uncertain=True)
        return self._save(row, 'MODIFY_FENCE')

    def cancel_started(self, oid, cause):
        row = self.rows()[str(oid)]
        if row['role'] != 'ENTRY':
            raise ValueError('Entry cancellation cannot touch protection')
        if row['cancel_started'] or row['closed']:
            return False
        row.update(cancel_started=True, cancel_cause=cause, cancel_at=self.clock().isoformat(),
                   phase='CANCEL_REQUESTED', uncertain=True)
        self._save(row, 'CANCEL_FENCE')
        return True

    def flat_exit_cancel_started(self,oid,cause,proof_sha256):
        row=self.rows()[str(oid)]
        if row['role'] not in {'STOP','TIME'} or len(proof_sha256)!=64:
            raise ValueError('Exact-owned verified-flat exit cleanup proof required')
        if row['cancel_started'] or row['closed']:return False
        row.update(cancel_started=True,cancel_cause=cause,cancel_at=self.clock().isoformat(),
                   flat_cleanup_proof_sha256=proof_sha256,phase='FLAT_EXIT_CANCEL_REQUESTED',uncertain=True)
        self._save(row,'FLAT_EXIT_CANCEL_FENCE')
        return True

    def native(self, event):
        oid = event.get('id'); row = self.rows().get(str(oid))
        if row is None:
            raise ValueError('Unjournaled native owner order')
        if event.get('source') != 'ibkr_wire' or any(event.get(k) != row[k] for k in
                ('account', 'client_id', 'session', 'con_id', 'ref')):
            raise ValueError('Native callback identity/provenance mismatch')
        perm = event.get('perm_id')
        if not isinstance(perm, int) or perm <= 0 or row['perm_id'] not in (0, perm):
            raise ValueError('Native permanent identity mismatch')
        row['perm_id'] = perm
        if event['kind'] == 'openOrder':
            # Current quantity/price echo is mandatory after every protection revision.
            if event.get('body_sha256') == row['body_sha256']:
                row['echo'] = deepcopy(event)
                row['echo_mismatch']=False
            else:
                row['echo']=None
                row['echo_mismatch']=True
                row['phase']='NATIVE_BODY_MISMATCH'
        elif event['kind'] == 'orderStatus':
            status = event['status']
            if status in TERMINAL:
                previous=row.get('terminal')
                if not previous or (event.get('filled') is not None and
                        (previous.get('status')!=status or previous.get('filled')!=event.get('filled'))):
                    row['terminal'] = deepcopy(event)
                row['phase'] = 'TERMINAL_AWAITING_EXECUTIONS'
            elif row['terminal'] is None:
                row['native'] = deepcopy(event)
                # Late Submitted cannot undo cancellation or resurrect a terminal order.
                if not row['cancel_started']:
                    row['phase'] = 'ACCEPTED' if status in ACCEPTED else 'BROKER_PENDING'
        row['uncertain'] = not self.accepted(row)
        return self._save(row, 'NATIVE_ORDER_EVIDENCE')

    def accepted(self, row):
        native = row.get('native') or {}
        return bool(row.get('echo') and row['echo'].get('status') in ACCEPTED and native.get('status') in ACCEPTED
                    and not row['cancel_started'] and not row['terminal'])

    def fill(self, oid, execution, qty, price):
        row = self.rows().get(str(oid))
        if row is None:
            raise ValueError('Unjournaled execution')
        if execution in row['fills']:
            return False
        stem = '.'.join(execution.split('.')[:3]) if len(execution.split('.')) >= 4 else execution
        if any(('.'.join(x.split('.')[:3]) if len(x.split('.')) >= 4 else x) == stem for x in row['fills']):
            raise ValueError('Execution correction requires explicit owner reconciliation')
        row['fills'][execution] = dict(qty=qty, price=price)
        row['filled'] += qty
        row['closed'] = False # even a late fill requires a new full read cycle
        self._save(row, 'LIFECYCLE_EXECUTION')
        return True

    def common(self, snapshot, fence, expected, journal_exec_ids):
        """Common exact-owner/history/generation proof, without exit coverage."""
        proof = snapshot.get('execution_health') or {}
        required = ('fresh_orders', 'fresh_positions', 'fresh_executions', 'fresh_completed')
        if (proof.get('source') != 'ibkr_responses' or proof.get('account') != self.account
                or proof.get('client_id') != self.client or proof.get('session') != self.session
                or not proof.get('connected') or not proof.get('healthy')
                or not all(proof.get(k) is True for k in required)
                or not proof.get('callback_stable') or fence != self.revision):
            return False, 'Incomplete, stale or overlapping owner response cycle'
        if proof.get('execution_since') != self.session or snapshot.get('own') is None:
            return False, 'Execution history does not cover the journal session'
        if set(journal_exec_ids) != set(snapshot.get('own_exec_ids') or []):
            return False, 'Missing or unjournaled owner execution identity'
        if any(qty for cid,qty in snapshot['own'].items() if cid not in expected):
            return False, 'Unknown owner contract exposure'
        if any(snapshot['own'].get(cid, 0) != qty for cid, qty in expected.items()):
            return False, 'Owner execution position mismatch'
        rows = self.rows()
        own_orders = [o for o in snapshot['orders'] if str(o.get('ref', '')).split('|')[2:3] == ['OpenBreakout']]
        for order in own_orders:
            row = rows.get(str(order['id']))
            if row is None or any(order.get(k) != row[k] for k in ('account', 'client_id', 'con_id', 'ref', 'perm_id')):
                return False, 'Unowned or mismatched OpenBreakout order'
        return True,None

    def reconcile(self, snapshot, fence, expected, journal_exec_ids):
        """A complete snapshot plus callbacks, never absence alone, proves closure."""
        valid,why=self.common(snapshot,fence,expected,journal_exec_ids)
        if not valid:return valid,why
        proof=snapshot['execution_health']
        rows=self.rows()
        own_orders=[o for o in snapshot['orders'] if str(o.get('ref','')).split('|')[2:3]==['OpenBreakout']]
        present = {o['id']: o for o in own_orders}
        complete_at = proof.get('started_at', '')
        for row in rows.values():
            if row['closed']:
                continue
            terminal = row.get('terminal')
            if terminal and terminal.get('received_at', '') < complete_at:
                if row['id'] not in present:
                    if terminal.get('filled') != row['filled'] or (terminal['status']=='Filled' and row['filled']!=row['body']['qty']):
                        return False, 'Terminal receipt quantity lacks matching executions'
                    row.update(closed=True, uncertain=False, phase='CLOSED')
                    self._save(row, 'ORDER_RECONCILED_CLOSED')
                    continue
            if not row['send_started'] or not row.get('echo') or row['id'] not in present:
                return False, 'Unbounded possible order exposure (omission is not closure)'
            order = present[row['id']]
            body = row['body']
            for k in ('kind', 'side', 'tif', 'oca', 'oca_type', 'stop', 'limit'):
                if k in body and order.get(k) != body[k]:
                    return False, 'Order body or protection revision mismatch'
            if order.get('remaining') != body['qty'] - row['filled']:
                return False, 'Remaining quantity mismatch'
            if row['cancel_started'] or row['terminal']:
                return False, 'Cancellation/terminal not reconciled'
            if row['role'] == 'ENTRY' and body.get('oca_type') != 1:
                return False, 'Two-sided entry exposure lacks OCA block proof'
            if row['role'] != 'ENTRY' and not self.accepted(row):
                return False, 'Current protection revision is not accepted'
            if row['role'] in {'STOP','TIME'}:
                net=expected.get(row['con_id'],0)
                if not net or body['side']!=(1 if net<0 else -1) or order['remaining']!=abs(net):
                    return False, 'Protection does not cover the current attributed position'
        return True, None


class OrderReliability:
    @property
    def reliability_enabled(self):
        return getattr(self.config, 'order_reliability_enabled', False)

    def reliability_init(self):
        self.lifecycle = None
        self.market_waits = {}
        self.ack_started = {}
        self.reliability_sleep = asyncio.sleep
        self.owner_cycle = None
        self.owner_snapshot = None
        self.owner_block = bool(self.store.get('reliability_owner_block', False))
        if self.reliability_enabled:
            self.reliability_policy = validate_policy(self.config.order_reliability_policy, self.config.mode)
            self.lifecycle = Lifecycle(self.store, self.config.account, self.config.client_id,
                                       self.manifest['day'], self.clock)
            self.broker.native_order_callback = self.reliability_native
            self.broker.order_error_policy = self.reliability_error_policy
            self.broker.lifecycle_revision = lambda:self.lifecycle.revision
            if hasattr(self.broker,'_intent_bodies'):
                self.broker._intent_bodies={int(k):dict(r['body']) for k,r in self.lifecycle.rows().items()}
            self.store.set('reliability_version', VERSION)
            self.store.set('reliability_qualification', self.reliability_policy)

    def reliability_native(self, event):
        if not self.reliability_enabled:
            return
        try:
            self.lifecycle.native(event)
            row=self.lifecycle.rows()[str(event['id'])]
            if row.get('echo_mismatch'):
                self.owner_block=True;self.store.set('reliability_owner_block',True)
                self.halt(f'ORDER_BODY_NATIVE_MISMATCH:{row["market"]}:{row["id"]}')
                self.loud(f'{row["market"]} NATIVE ORDER BODY MISMATCH; current acceptance unverified; CHECK TWS')
            if row['role'] in {'STOP','TIME'} and event.get('status') in {'Cancelled','ApiCancelled','Inactive'} and not row['cancel_started']:
                self.owner_block=True;self.store.set('reliability_owner_block',True)
                self._spawn(self.reliability_protection_check(row['market']))
        except Exception as exc:
            self.halt(f'ORDER_NATIVE_PROOF:{exc}')
            return False
        return True

    def initiating_cause(self, reason):
        if not self.reliability_enabled:
            return str(reason)
        first = self.store.get('initiating_cause')
        if first is None:
            first = dict(id=hashlib.sha256(f'{self.manifest["day"]}:{self.clock().isoformat()}:{reason}'.encode()).hexdigest(),
                         reason=str(reason), received_at=self.clock().isoformat())
            self.store.set('initiating_cause', first)
            self.store.event('INITIATING_CAUSE', first)
        elif first['reason'] != str(reason):
            self.store.event('LINKED_CONSEQUENCE', dict(cause_id=first['id'], reason=str(reason),
                                                       received_at=self.clock().isoformat()))
        return first['reason']

    async def reliability_snapshot(self):
        fence = self.lifecycle.revision
        snapshot = await asyncio.wait_for(self.broker.snapshot(), self.reliability_policy['snapshot_seconds'])
        self.owner_snapshot=snapshot
        # A native response legitimately increments revision. Snapshot records its
        # post-read fence; any execution or protection update after it invalidates it.
        proof = snapshot.get('execution_health') or {}
        received=datetime.fromisoformat(proof['received_at']) if proof.get('received_at') else None
        fresh=bool(received and received.tzinfo and 0 <= (self.clock()-received).total_seconds() <= self.reliability_policy['snapshot_seconds']
                   and 0 <= proof.get('roundtrip_seconds',-1) <= self.reliability_policy['snapshot_seconds'])
        response_fence = proof.get('lifecycle_revision', fence)
        expected = self._expected()
        from .service import exec_key
        journal_ids=[exec_key(e) for e in self.store.fill_ids()]
        common_safe,common_why=self.lifecycle.common(snapshot,response_fence,expected,journal_ids)
        safe, why = self.lifecycle.reconcile(snapshot, response_fence, expected,journal_ids)
        if not fresh:
            safe,why=False,'Owner response cycle is stale or outside qualified read deadline'
        for state in self.states.values():
            if state.qty and state.market not in self.manual_markets:
                if not self.protection_current(state,snapshot):
                    safe,why=False,'Current held position lacks exact accepted protective stop'
                if state.phase=='OPEN':
                    row=self.lifecycle.rows().get(str(state.time_order))
                    order=next((o for o in snapshot['orders'] if o['id']==state.time_order and o.get('client_id')==self.config.client_id),None)
                    if not row or not order or not self.lifecycle.accepted(row) or order.get('remaining')!=state.qty:
                        safe,why=False,'Open position lacks exact accepted timed exit'
        self.owner_cycle = dict(safe=safe, why=why, revision=self.lifecycle.revision,
                                common_safe=bool(common_safe and fresh),common_why=common_why,
                                at=self.monotonic(), session=self.manifest['day'])
        self.owner_block = not safe
        self.store.set('reliability_owner_block', self.owner_block)
        self.store.event('OWNER_RESPONSE_CYCLE', self.owner_cycle)
        return safe

    def reliability_error_policy(self,oid,code,message):
        row=self.lifecycle.rows().get(str(oid))
        if not row or row['role']!='ENTRY' or not row['cancel_started'] or not row['perm_id']:
            return None
        native=row.get('native') or {};terminal=row.get('terminal') or {}
        exact_cancel_race=(code==10148 and native.get('status')=='PendingCancel')
        oca_consequence=(code==201 and 'oca order cancel' in str(message).lower()
                         and terminal.get('status')=='Inactive' and row['body'].get('oca_type')==1)
        if not exact_cancel_race and not oca_consequence:return None
        # Quarantine until complete executions/closure. This is an order-state
        # consequence, not proof that the connection failed or that we're flat.
        self.owner_block=True;self.store.set('reliability_owner_block',True)
        self.store.event('CANCEL_ERROR_QUARANTINED',dict(id=oid,code=code,
                        cause_id=row['cancel_cause'],closure_confirmed=row['closed']))
        return 'cancel_consequence_confirmed' if row['closed'] else 'cancel_consequence_unreconciled'

    def protection_current(self,state,snapshot):
        row=self.lifecycle.rows().get(str(state.stop_order))
        order=next((o for o in snapshot['orders'] if o['id']==state.stop_order
                    and o.get('client_id')==self.config.client_id),None)
        return bool(row and order and self.lifecycle.accepted(row)
                    and all(order.get(k)==row[k] for k in ('account','client_id','con_id','ref','perm_id'))
                    and order.get('kind')=='STP' and order.get('side')==-state.side
                    and order.get('remaining')==state.qty and order.get('stop')==state.stop
                    and order.get('tif')=='GTC' and order.get('oca')==state.oca)

    async def reliability_protection_check(self,market):
        try:
            await self.reliability_snapshot()
            state=self.states[market]
            if state.qty and not self.protection_current(state,self.owner_snapshot):
                self.halt(f'PROTECTIVE_NATIVE_TERMINAL:{market}:{state.stop_order}')
                self.loud(f'{market} PROTECTION UNVERIFIED: exact native stop acceptance/quantity missing; CHECK TWS')
        except Exception as exc:
            self.halt(f'PROTECTION_RECONCILIATION:{market}:{exc}')

    async def reliability_watchdog(self):
        safe=await self.reliability_snapshot()
        snapshot=self.owner_snapshot
        self.reliability_exit_cleanup_check()
        if not safe and self.owner_cycle.get('why')=='Remaining quantity mismatch':
            self.halt('PROTECTIVE_OCA_REDUCTION_REQUIRES_QUALIFIED_RECONCILIATION')
            self.loud('OpenBreakout OCA quantity changed: current revision proof incomplete; CHECK TWS')
        for state in self.states.values():
            if state.market in self.manual_markets:continue
            if state.qty and not self.protection_current(state,snapshot):
                row=self.lifecycle.rows().get(str(state.stop_order))
                pending=bool(row and not row['terminal'] and not row['cancel_started']
                             and row['body']['qty']-row['filled']==state.qty and row['body'].get('stop')==state.stop
                             and self.monotonic()-self.ack_started.get(state.stop_order,float('-inf'))
                             < self.reliability_policy['protection_ack_seconds'])
                if not pending:
                    self.halt(f'PROTECTION_UNVERIFIED:{state.market}:{state.stop_order}')
                    self.loud(f'{state.market} PROTECTIVE STOP UNVERIFIED; CHECK TWS')
            if state.qty and state.phase=='OPEN':
                timed=self.lifecycle.rows().get(str(state.time_order))
                if not timed or not self.lifecycle.accepted(timed):
                    self.halt(f'TIMED_EXIT_UNVERIFIED:{state.market}:{state.time_order}')
            local=self.clock().astimezone(NY)
            if local.time()>=time(15,56) and state.qty:
                self.halt(f'TIMED_EXIT_INCOMPLETE:{state.market}')
            if not self.price_pause_enabled and state.opening and local.time()<time(11,30) and state.last_timestamp:
                from .strategy import aware
                if (self.clock()-aware(state.last_timestamp)).total_seconds()>self.config.watchdog_stale_seconds:
                    self.halt('RECONCILE:Signal stream stale')
            self.store.save(state)
        if self.clock().astimezone(NY).time()>=time(15,56) and snapshot['orders']:
            if any(str(o.get('ref','')).split('|')[2:3]==['OpenBreakout'] for o in snapshot['orders']):
                self.halt('OWN_ORDER_WORKING_AFTER_TIMED_EXIT_DEADLINE')
        if self.price_pause_enabled and safe:
            self._update_price_health(snapshot,snapshot['own'],self._expected(),[],True)
        self.store.set('heartbeat',dict(at=self.clock().isoformat(),halted=self.halted,
                         positions=self._expected(),own_known=snapshot.get('own') is not None,
                         order_reliability=self.reliability_health()))
        return safe

    async def reliability_flat_exit_cleanup(self,market):
        """An unexpected net-zero OCA fill cannot leave reversing exits behind.

        Entry closure, complete exact-owner executions/positions and unchanged
        callback revision are required before the distinct exit cleanup fence.
        Any ambiguity keeps a loud operator incident, never broad cancellation.
        """
        try:
            await self.cancel_resting_entries(market)
            await self.reliability_snapshot()
            state=self.states[market];snapshot=self.owner_snapshot
            entries=state.entry_orders or ([state.entry_order] if state.entry_order else [])
            if state.qty or any(not self.lifecycle.rows()[str(o)]['closed'] for o in entries):
                raise ValueError('Entries/net position are not reconciled flat')
            # Ignore exit coverage only for already-proved flat cleanup. The
            # same full source/identity/history/freshness proof is mandatory.
            protected_rows=self.lifecycle.rows()
            if (not self.owner_cycle.get('common_safe') or self.owner_cycle['revision']!=self.lifecycle.revision
                    or snapshot['own'].get(self.markets[market].execution.con_id,0)!=0):
                raise ValueError('Flat exit cleanup owner proof is incomplete')
            proof_hash=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
            ids=[o for o in (state.stop_order,state.time_order) if o]
            cleanup=self.store.get('flat_exit_cleanup',{})
            previous=cleanup.get(market)
            if previous and previous['ids']!=ids:
                raise ValueError('Unclosed cleanup identities changed; manual reconciliation required')
            if previous is None:
                cleanup[market]=dict(ids=ids,started_at=self.clock().isoformat(),proof_sha256=proof_hash)
            self.store.set('flat_exit_cleanup',cleanup)
            revision=self.lifecycle.revision
            execution_ids=set(self.store.fill_ids())
            for oid in ids:
                if not oid:continue
                if state.qty or revision!=self.lifecycle.revision:
                    raise ValueError('Execution or callback invalidated flat cleanup proof')
                if any(r.get('echo_mismatch') for r in self.lifecycle.rows().values()):
                    raise ValueError('Current native body mismatch invalidates flat cleanup proof')
                row=self.lifecycle.rows()[str(oid)]
                order=next((o for o in snapshot['orders'] if o['id']==oid and o.get('client_id')==self.config.client_id),None)
                if row['closed']:continue
                if row['cancel_started']:continue # never resend a consumed fence
                if not order or not self.lifecycle.accepted(row) or any(order.get(k)!=row[k] for k in ('account','client_id','con_id','ref','perm_id')):
                    raise ValueError('Flat exit cleanup lacks exact owned accepted exit')
                first=self.store.get('initiating_cause') or {}
                if self.lifecycle.flat_exit_cancel_started(oid,first.get('id','NORMAL_EXIT_FLAT'),proof_hash):
                    self.broker.cancel(oid)
                    revision=self.lifecycle.revision
                    # The synchronous adapter may deliver an execution while
                    # cancelling. Never reuse flat proof for another exit.
                    if state.qty or set(self.store.fill_ids())!=execution_ids:
                        raise ValueError('Flat position changed during exit cancellation')
            state.phase='BLOCKED';state.long_armed=state.short_armed=False;self.store.save(state)
            await self.reliability_snapshot()
            self.reliability_exit_cleanup_check()
        except Exception as exc:
            self.halt(f'FLAT_EXIT_CLEANUP_UNCONFIRMED:{market}:{exc}')
            self.loud(f'{market} FLAT EXIT CLEANUP UNCONFIRMED; stale exits may reverse; CHECK TWS')

    def reliability_exit_cleanup_check(self):
        """Persisted closure/deadline survives restart and never retries cancels."""
        cleanup=self.store.get('flat_exit_cleanup',{})
        for market,item in list(cleanup.items()):
            rows=self.lifecycle.rows()
            pending=[oid for oid in item['ids'] if not rows.get(str(oid),{}).get('closed')]
            if not pending:
                self.store.event('FLAT_EXIT_CLEANUP_CLOSED',dict(market=market,ids=item['ids']))
                del cleanup[market]
                continue
            self.owner_block=True;self.store.set('reliability_owner_block',True)
            self.store.event('FLAT_EXIT_CLEANUP_PENDING',dict(market=market,ids=pending))
            if (self.clock()-datetime.fromisoformat(item['started_at'])).total_seconds()>=self.reliability_policy['cancel_seconds']:
                self.halt(f'FLAT_EXIT_CANCEL_RECONCILIATION_TIMEOUT:{market}:{pending}')
                self.loud(f'{market} EXIT CANCELLATION UNCONFIRMED AFTER FLAT; stale exits may reverse; CHECK TWS')
        self.store.set('flat_exit_cleanup',cleanup)

    def reliability_health(self):
        first=self.store.get('initiating_cause')
        return dict(version=VERSION,enabled=self.reliability_enabled,
                    qualification=self.store.get('reliability_qualification'),
                    initiating_cause=first,owner_admission_blocked=self.owner_block,
                    owner_cycle=self.owner_cycle,market_waits=dict(self.market_waits),
                    native_acceptance_proof='wire_callback_and_current_openOrder_echo',
                    restart_resends=False,coordination_automatic_close=False)

    def reliability_admission(self):
        cycle = self.owner_cycle or {}
        return bool(not self.halted and not self.owner_block and cycle.get('safe') and cycle.get('revision') == self.lifecycle.revision
                    and self.monotonic() - cycle.get('at', 0) <= self.reliability_policy['snapshot_seconds'])

    async def reliability_wait(self, oid):
        """Separate per-market ACK grace; SDK PendingSubmit is not acceptance."""
        started = self.ack_started.get(oid,self.monotonic())
        soft = self.reliability_policy['soft_ack_seconds']
        hard = self.reliability_policy['hard_ack_seconds']
        if self.lifecycle.rows()[str(oid)]['role']!='ENTRY':
            hard=self.reliability_policy['protection_ack_seconds']
        self.market_waits[oid] = 'AWAITING_NATIVE'
        while True:
            row = self.lifecycle.rows()[str(oid)]
            if self.lifecycle.accepted(row):
                self.market_waits.pop(oid, None)
                return
            if row['terminal'] or row['filled'] or row['cancel_started']:
                await self.reliability_snapshot()
                if row['terminal'] and not row['cancel_started'] and not row['filled'] and not self.states[row['market']].qty:
                    self.halt(f'UNEXPECTED_NATIVE_TERMINAL:{row["market"]}:{oid}')
                self.market_waits.pop(oid, None)
                return
            elapsed = self.monotonic() - started
            if elapsed >= soft:
                self.market_waits[oid] = 'RECONCILING_ACK'
                await self.reliability_snapshot()
            if elapsed >= hard:
                self.halt(f'ORDER_ACCEPTANCE_TIMEOUT:{row["market"]}:{oid}')
                self.market_waits.pop(oid, None)
                return
            await self.reliability_sleep(min(.05, hard - elapsed))

    async def reliability_cancel_entries(self, market=None):
        for s in self.states.values():
            if (market is not None and market != s.market) or s.market in self.manual_markets:
                continue
            pending = []
            for oid in s.entry_orders:
                row = self.lifecycle.rows().get(str(oid))
                if row is None:
                    self.halt(f'ORDER_CANCEL_WITHOUT_INTENT:{oid}'); continue
                if row['closed']:
                    continue
                if not row['send_started']:
                    # The durable pre-wire fence is absent: this intent was never sent.
                    row.update(closed=True, uncertain=False, phase='UNSENT_CLOSED')
                    self.lifecycle._save(row,'UNSENT_INTENT_CLOSED')
                    continue
                pending.append(oid)
                cause = self.store.get('initiating_cause') or dict(id='ENTRY_REMAINDER_OR_EXPIRY')
                # Persist first; ambiguous crash gaps never trigger automatic resend.
                if self.lifecycle.cancel_started(oid, cause['id']):
                    self.cancel_requested.add(oid)
                    try:
                        self.broker.cancel(oid)
                    except Exception as exc:
                        self.store.event('ENTRY_CANCEL_FAILED', dict(id=oid, error=str(exc)))
            if pending:
                try:
                    await self.reliability_snapshot()
                except Exception as exc:
                    self.owner_block = True
                    self.store.set('reliability_owner_block', True)
                    self.store.event('CANCEL_RECONCILIATION_FAILED', dict(error=str(exc)))
                if any(not self.lifecycle.rows()[str(oid)]['closed'] for oid in pending):
                    self.owner_block = True
                    self.store.set('reliability_owner_block', True)
                    self.store.event('CANCEL_PENDING_OWNER_RECONCILIATION', dict(ids=pending))
                    for oid in pending:
                        row=self.lifecycle.rows()[str(oid)]
                        age=(self.clock()-datetime.fromisoformat(row['cancel_at'])).total_seconds() if row.get('cancel_at') else 0
                        if not row['closed'] and age >= self.reliability_policy['cancel_seconds']:
                            self.halt(f'ENTRY_CANCEL_RECONCILIATION_TIMEOUT:{s.market}:{oid}')
            if pending and not s.qty and all(self.lifecycle.rows()[str(oid)]['closed'] for oid in pending):
                s.phase='BLOCKED'; s.long_armed=s.short_armed=False; self.store.save(s)

    async def reliable_park_entries(self, s):
        """Only atomic reservation/send fencing shares a lock; all waits are outside."""
        try:
            m = self.markets[s.market]
            equity = await asyncio.wait_for(self.broker.equity(), 10.)
            # A pending peer can be isolated only after complete exact-owner proof.
            if not self.reliability_admission():
                await self.reliability_snapshot()
            if not self.reliability_admission():
                s.phase='BLOCKED'; s.note='ACCOUNT_ORDER_EXPOSURE_UNKNOWN'
                self.store.save(s); return
            bid, ask = self.fresh_quote(s.market)
            candidates = []
            for side, armed in ((1,s.long_armed),(-1,s.short_armed and s.score>=20)):
                trigger = snap(s.opening+side*s.distance, m.execution.tick, side==1)
                inside = max(s.previous,ask)<trigger if side==1 else min(s.previous,bid)>trigger
                if armed and inside and self._coord_allowed(s.market, side):
                    candidates.append({**size_order(s,m,side,trigger,trigger,equity,self.config), 'trigger':trigger})
            if self.config.mode != 'live':
                for plan in candidates:
                    await asyncio.wait_for(self.broker.check_margin(m,plan,equity),10.)
            async with self.entry_lock:
                if self.halted or self.price_entry_blocked or s.market in self.manual_markets or not self.reliability_admission():
                    s.phase='BLOCKED';self.store.save(s);return
                if self.clock().astimezone(NY).time() >= time(11,30):
                    s.phase='BLOCKED';self.store.save(s);return
                # Re-read risk and fresh boundaries inside the atomic admission fence.
                daily = self.store.get('daily_reserved',0.)
                opened = sum(x.planned_risk for x in self.states.values())
                room = min(equity*self.config.max_daily_risk_bps/10000-daily,
                           equity*self.config.max_open_risk_bps/10000-opened)
                bid,ask = self.fresh_quote(s.market)
                plans=[]
                for p in candidates:
                    if not self._coord_allowed(s.market,p['side']) or not (max(s.previous,ask)<p['trigger'] if p['side']==1 else min(s.previous,bid)>p['trigger']):
                        continue
                    qty=min(p['qty'],max(0,math.floor(room/p['per_contract']+1e-9)),self.margin_day_cap.get(s.market,p['qty']))
                    if qty: plans.append({**p,'qty':qty,'risk':qty*p['per_contract']})
                if not plans:
                    s.phase='FLAT';self.store.save(s);return
                risk=max(p['risk'] for p in plans)
                group=f'OB-ENTRY-{self.config.client_id}-{s.day}-{s.market}-{s.attempts+1}'
                sends=[]
                with self.store.transaction():
                    s.attempts+=1;s.phase='RESTING';s.entry_orders=[];s.entry_risks={}
                    s.entry_order=s.stop_order=s.time_order=0;s.side=0;s.planned_risk=risk
                    s.long_armed=s.short_armed=False
                    self.store.set('daily_reserved',daily+risk)
                    for p in plans:
                        oid=self.broker.next_id()
                        body=dict(kind='STP LMT',side=p['side'],qty=p['qty'],stop=p['trigger'],limit=p['limit'],
                                  tif='GTD',good_till=f'{s.day.replace("-","")} 11:30:00 America/New_York',
                                  trigger_method=2,oca=group,oca_type=1,account=self.config.account,
                                  ref=f'{m.execution.symbol}|{"BUY" if p["side"]==1 else "SELL"}|OpenBreakout|{s.day}|{s.market}-{s.attempts}-ENTRY')
                        s.entry_orders.append(oid);s.entry_risks[str(oid)]=p['risk']
                        self.store.order(oid,s.market,'ENTRY',body)
                        self.lifecycle.intent(oid,s.market,'ENTRY',m.execution.con_id,body,p['risk'])
                        sends.append((oid,body))
                    self.store.save(s)
                # Send is synchronous; no equity, margin, ACK or cancellation awaits
                # occur under the shared risk lock. All intents precede first send.
                for oid,body in sends:
                    if s.qty or self.halted or self.price_entry_blocked:
                        break
                    self.lifecycle.send_started(oid)
                    self.ack_started[oid]=self.monotonic()
                    self._coord_entry_send(oid,m,body)
            await asyncio.gather(*(self.reliability_wait(oid) for oid,_ in sends
                                   if self.lifecycle.rows()[str(oid)]['send_started']))
        except Exception as exc:
            self.halt(f'RELIABLE_RESTING_ENTRY:{s.market}:{exc}')
