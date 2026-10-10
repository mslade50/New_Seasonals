"""Candidate hooks. Inert unless explicitly enabled or injected in offline tests."""
import asyncio
from intraday_coordination import CoordinationError, EntryBlocked, Scope, from_environment


class CoordinationHooks:
    def _coord_init(self, coordination=None, coordination_backend=None):
        self.coordination = coordination if coordination is not None else from_environment()
        self.coordination_backend = coordination_backend
        if self.coordination and self.coordination_backend is None and hasattr(self.broker, 'ib') and hasattr(self.broker, 'snapshot_lock'):
            from .native_coordination import NativeOwnerAdapter, AsyncOwnerCoordinator
            self.coordination_backend = AsyncOwnerCoordinator(NativeOwnerAdapter(self))
        self.coord_poll_queued = False
        self.coord_poll_lock = asyncio.Lock()
        self.coord_reconcile = set(self.store.get('coord_reconcile', []))

    def _coord_scope(self, market):
        return Scope.for_symbol(self.states[market].day, self.config.account,
                                self.markets[market].execution.symbol)

    def _coord_allowed(self, market, side):
        # A reconciliation pause is a safety gate independent of directional eligibility.
        return (market not in self.coord_reconcile
                and (not self.coordination or self.coordination.allowed(self._coord_scope(market), side)))

    def _coord_filter_arming(self, state):
        before=(state.long_armed,state.short_armed)
        if not self._coord_allowed(state.market,1):state.long_armed=False
        if not self._coord_allowed(state.market,-1):state.short_armed=False
        return before!=(state.long_armed,state.short_armed)

    def _coord_entry_send(self, oid, market, body):
        def transmit():
            # Entry-only: execution-driven stop/time orders never use this gate.
            if (self.halted or market.name in self.manual_markets or market.name in self.coord_reconcile
                    or self.price_entry_blocked):
                raise EntryBlocked('Owner hard/manual/reconciliation/price gate blocks new entry transmission')
            return self.broker.send(oid, market, body)
        try:
            if not self.coordination:
                return transmit()
            parts = str(body.get('ref') or '').split('|')
            scope = self._coord_scope(market.name)
            if (len(parts) < 4 or parts[2] != 'OpenBreakout' or parts[3] != scope.day
                    or parts[0] != market.execution.symbol or body.get('account') != scope.account):
                raise CoordinationError('Entry identity differs from its strategy/account/session/contract')
            token = f'OB:{self.config.client_id}:{oid}'
            identity = dict(strategy='OpenBreakout', role='ENTRY', account=scope.account,
                            con_id=market.execution.con_id, client_id=self.config.client_id,
                            order_id=oid, perm_id=0)
            if (self.halted or market.name in self.manual_markets or market.name in self.coord_reconcile
                    or self.price_entry_blocked):
                raise EntryBlocked('Owner/price gate blocks preparation of a new durable entry')
            return self.coordination.submit_entry(scope, body['side'], token, identity,
                transmit)
        except EntryBlocked:
            self.store.order_status(oid, 'Cancelled')  # this local PREPARED intent was never transmitted
            self.store.event('COORD_ENTRY_SUPPRESSED', dict(market=market.name, side=body['side'], id=oid))
            raise

    def _coord_queue_poll(self):
        if self.coordination and not self.coord_poll_queued and not self.coord_poll_lock.locked():
            self.coord_poll_queued = True
            self._spawn(self._coord_poll())

    async def _coord_poll(self):
        # One poll at a time. Watchdog and tick callbacks do not build an await queue.
        if self.coord_poll_lock.locked():
            return
        async with self.coord_poll_lock:
            await self._coord_poll_once()

    async def _coord_poll_once(self):
        try:
            for name, state in self.states.items():
                scope = self._coord_scope(name)
                body = self.coordination.read(scope)
                if not body['signal']:
                    continue
                if self.halted and self.coordination_backend is None:
                    self._coord_flag(name, 'Owner hard halt; automatic coordination mutations blocked; protection retained')
                    continue
                if self.coordination_backend is not None:
                    # Native reads await on the existing owner connection before
                    # the short durable/send critical section. Default filled
                    # closes preserve protection and require manual review.
                    phase = await self.coordination_backend.poll(scope, self.clock())
                    if phase == 'reconcile':
                        self._coord_flag(name, self.coordination.read(scope)['operation']['reason'])
                    # A native OCA cancellation can also remove the eligible sibling.
                    # Release only local open-risk reservation after exact fresh terminal
                    # receipts, preserving attempts and consumed daily risk.
                    snap=getattr(getattr(self.coordination_backend,'adapter',None),'cache',None)
                    if (phase in {'cleanup','done'} and state.phase=='RESTING' and not state.qty
                            and snap and snap.scope==scope and snap.complete and snap.ownership_known
                            and not snap.manual_control and not snap.owned_qty
                            and not any(v.strategy=='OpenBreakout' and v.role=='ENTRY' and v.active for v in snap.orders)):
                        state.phase='FLAT';state.planned_risk=0.
                        state.long_armed=not self.price_paused and self._coord_allowed(name,1)
                        state.short_armed=not self.price_paused and self._coord_allowed(name,-1)
                        self.store.save(state)
                    continue
                if name in self.manual_markets:
                    self._coord_flag(name, 'Operator controls market; no automatic mutations')
                    continue
                opposite = -body['signal']['side']
                orders = self.store.orders()
                pending = [oid for oid, order in orders.items() if order['market'] == name
                           and order['role'] == 'ENTRY' and order['body']['side'] == opposite
                           and order['body'].get('account') == scope.account
                           and order['status'] not in {'Cancelled', 'ApiCancelled', 'Filled', 'Inactive', 'REJECTED'}]
                # Mark same-group sibling cancellation as expected: the broker
                # may cancel it via OCA. It remains eligible for subsequent repark.
                groups = {orders[oid]['body'].get('oca') for oid in pending}
                for oid, order in orders.items():
                    if order['market'] == name and order['role'] == 'ENTRY' and order['body'].get('oca') in groups:
                        self.cancel_requested.add(oid)
                for oid in pending:
                    self.cancel_requested.add(oid)
                    self.store.event('COORD_CANCEL_INTENT', dict(id=oid, market=name, side=opposite))
                    try:
                        self.broker.cancel(oid)
                    except Exception:
                        self._coord_flag(name, 'Entry cancellation delivery unknown')
                        continue
                    if not await self._await_cancelled(oid, self.cancel_timeout) and self.broker.status(oid) != 'Filled':
                        self._coord_flag(name, 'Entry cancellation not confirmed')
                if state.qty and state.side == opposite:
                    self._coord_flag(name, 'Owned filled position requires qualified close backend; protection retained')
                elif not state.qty and state.phase == 'RESTING' and not self.working_entries(state):
                    state.phase = 'FLAT'
                    state.planned_risk = 0.
                    state.long_armed = not self.price_paused and self._coord_allowed(name, 1)
                    state.short_armed = not self.price_paused and self._coord_allowed(name, -1)
                    self.store.save(state)
        except Exception as exc:
            # Storage/unknown state is a safety pause, not a directional policy.
            self.halt(f'COORDINATION_STATE_UNKNOWN:{exc}')
        finally:
            self.coord_poll_queued = False

    def _coord_flag(self, market, reason):
        self.coord_reconcile.add(market)
        self.store.set('coord_reconcile', sorted(self.coord_reconcile))
        self.store.event('COORD_RECONCILE_REQUIRED', dict(market=market, reason=reason))
        self.loud(f'{market} coordination requires reconciliation: {reason}')
