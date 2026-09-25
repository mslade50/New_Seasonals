"""Journal-first lifecycle. Broker interface is also implemented by the offline replay."""
import asyncio
from datetime import datetime, time, timezone
import math
from .strategy import State, NY, aware, on_price, size_order, snap

TERMINAL = {'Filled','Cancelled','ApiCancelled','Inactive','REJECTED'}
ACKNOWLEDGED = {'PreSubmitted','Submitted','Filled'}
# Explicit broker rejection of a protective order; warning-class codes never flatten.
REJECT_CODES = {110,200,201,203,321,10147}
WARNING_CODES = {2104,2106,2108,2109,2158,399,404}
# A cancel request answered with these codes did not cancel the order.
NOT_CANCELLED_CODES = {161,10148}
STOP_DEAD = {'Inactive','Cancelled','ApiCancelled','REJECTED'}
CANCELLED = {'Cancelled','ApiCancelled','Inactive'}
STRATEGY_REF = 'OpenBreakout'
MISMATCH_CHECKS = 3

class Service:
    def __init__(self, config, manifest, store, broker, clock=None):
        self.config,self.manifest,self.store,self.broker = config,manifest,store,broker
        self.clock = clock or (lambda:datetime.now(timezone.utc))
        self.markets = {m.name:m for m in config.markets}
        old = store.states()
        self.states = {s['market']:State(**s) for s in old} if old else {
            m.name:State(manifest['day'],m.name,manifest['markets'][m.name]['prior_tr'],manifest['score']) for m in config.markets}
        self.tasks = set()
        self.entry_lock = asyncio.Lock()
        self.notify = lambda text:None
        self.alerted = set()
        self.flatten_delay = 1.
        self.cancel_timeout = 3.
        self.sibling_timeout = 10.
        self.mismatches = 0
        self.cancel_requested = set()
        self.unexpected = set()
        self.flattened = set(store.get('flattened',[]))
        self.halted = bool(store.get('halted',False))
        if store.get('manifest_hash',manifest['hash']) != manifest['hash']:
            raise ValueError('Manifest changed for existing journal')
        store.set('manifest_hash',manifest['hash'])
        if store.orders():
            self.halt('RESTART_REQUIRES_READ_ONLY_RECONCILIATION')
        self._apply_prior_range()
        for state in self.states.values(): store.save(state)
        broker.fill_callback = self.fill
        broker.status_callback = self.status
        broker.halt_callback = self.halt
        broker.order_error_callback = self.order_error

    def _apply_prior_range(self):
        """A skipped market is never armed: no opening capture, no entries, no missed-open block."""
        filt = self.config.prior_range_filter
        for name,s in self.states.items():
            item = self.manifest['markets'][name]
            # Fail closed if an enabled filter meets a manifest without an OK decision.
            skip = item.get('skip_prior_range',False) is True or (self.config.range_filter_on and item.get('prior_range_status')!='OK')
            if item.get('half_prior_range') is True and not s.half_size and not s.attempts:
                s.half_size = True
                self.store.event('PRIOR_RANGE_HALF',dict(market=name,ratio=item.get('ratio'),atr20=item.get('atr20'),
                                                         threshold=filt.threshold if filt else None))
            if not skip or s.phase=='SKIPPED' or s.qty or s.attempts or s.opening:
                continue
            s.phase='SKIPPED'
            s.long_armed=s.short_armed=False
            s.note=f'PRIOR_RANGE_SKIP: {item.get("prior_range_reason") or "filter enabled without an OK prior-range decision"}'
            self.store.event('PRIOR_RANGE_SKIP',dict(market=name,ratio=item.get('ratio'),atr20=item.get('atr20'),
                                                     prior_tr=item.get('prior_tr'),status=item.get('prior_range_status'),
                                                     threshold=filt.threshold if filt else None,reason=s.note))
            print(f'{name} not armed: {s.note}',flush=True)

    def halt(self, reason):
        self.halted = True
        self.store.set('halted',True)
        if self.store.get('halt_reason') != str(reason):
            self.store.event('HALT',{'reason':str(reason)})
            self.store.set('halt_reason',str(reason))
        held = {s.market:s.side*s.qty for s in self.states.values() if s.qty}
        # Alert every new reason, and again when a reason recurs with a position open.
        key = (str(reason),bool(held))
        if key not in self.alerted:
            self.alerted.add(key)
            self.notify(f'HALTED ({self.config.mode}){" WITH OPEN POSITION "+str(held) if held else ""}: {reason}')
        # Does not cancel protection or flatten, except the explicit stop-rejection path.

    def loud(self, msg):
        print(f'!!! {msg}',flush=True);self.notify(msg)

    def _spawn(self, coro):
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close();return None
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def _ref(self, state, role, attempt=None):
        # Pipe format parsed by daily_execution_report.parse_ref: strategy is the 3rd field.
        action = 'BUY' if state.side == 1 else 'SELL'
        symbol = self.markets[state.market].execution.symbol
        return f'{symbol}|{action}|{STRATEGY_REF}|{state.day}|{state.market}-{state.attempts if attempt is None else attempt}-{role}'

    def status(self, order_id, status):
        orders = self.store.orders()
        if order_id in orders:
            self.store.order_status(order_id,status)
            self.store.event('ORDER_STATUS',dict(id=order_id,status=status))
            if status in {'Inactive','REJECTED'}:
                self.halt(f'ORDER_REJECTED:{order_id}')
            if status in STOP_DEAD and orders[order_id]['role'] == 'STOP':
                # ib_insync synthesizes Cancelled for any non-warning error, so only a genuine
                # Inactive or a rejection-class error code counts as an explicit rejection.
                code = self.broker.last_error_code(order_id)
                explicit = status in {'Inactive','REJECTED'} or code in REJECT_CODES
                self._stop_dead(orders[order_id]['market'], order_id, f'STOP_{status}:code={code}', explicit)

    def order_error(self, order_id, code, message):
        orders = self.store.orders()
        if order_id not in orders:
            return
        self.store.event('ORDER_ERROR',dict(id=order_id,code=code,message=str(message)))
        if code in WARNING_CODES or code not in REJECT_CODES:
            return
        if orders[order_id]['role'] == 'STOP':
            self._stop_dead(orders[order_id]['market'], order_id, f'STOP_ERROR_{code}', True)

    def _stop_dead(self, market, order_id, reason, explicit):
        s = self.states[market]
        # Our own cancel requests and stale stop IDs are not rejections.
        if order_id in self.cancel_requested or order_id != s.stop_order or not s.qty:
            return
        if not explicit:
            if order_id not in self.unexpected:
                self.unexpected.add(order_id)
                if not self._spawn(self._unexpected_stop_cancel(market, order_id, reason)):
                    self.halt(f'PROTECTIVE_STOP_CANCELLED:{market}:{order_id}:{reason}')
                    self.loud(f'{market} PROTECTIVE STOP CANCELLED ({reason}); NO AUTO-FLATTEN; CHECK TWS, FLATTEN BY HAND')
            return
        if market in self.flattened:
            return
        self.flattened.add(market)
        self.store.set('flattened',sorted(self.flattened))
        self.halt(f'PROTECTIVE_STOP_REJECTED:{market}:{order_id}:{reason}')
        print(f'!!! {market} PROTECTIVE STOP REJECTED ({reason}); emergency flatten check in progress',flush=True)
        if not self._spawn(self._emergency_flatten(market, reason)):
            self.halt(f'EMERGENCY_FLATTEN_NOT_SCHEDULED:{market}')
            self.loud(f'{market} EMERGENCY FLATTEN COULD NOT BE SCHEDULED; FLATTEN BY HAND IN TWS')

    async def _unexpected_stop_cancel(self, market, order_id, reason):
        # Give an OCA sibling fill (timed exit) time to reconcile before alarming.
        await asyncio.sleep(self.flatten_delay)
        s = self.states[market]
        if s.qty and s.stop_order == order_id:
            self.halt(f'PROTECTIVE_STOP_CANCELLED:{market}:{order_id}:{reason}')
            self.loud(f'{market} PROTECTIVE STOP CANCELLED WITHOUT AN EXPLICIT REJECTION ({reason}); '
                      f'NO AUTO-FLATTEN; CHECK TWS, FLATTEN BY HAND IF UNPROTECTED')

    async def _await_cancelled(self, oid, timeout):
        loop = asyncio.get_running_loop()
        deadline = loop.time()+timeout
        while True:
            status = self.broker.status(oid)
            if status == 'Filled':
                return False
            if status in CANCELLED and self.broker.last_error_code(oid) not in NOT_CANCELLED_CODES:
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(.05)

    async def _emergency_flatten(self, market, reason):
        """At most one market flatten per market/day, only after TWS confirms both exits are cancelled
        and the broker still holds the journal position."""
        s = self.states[market]
        m = self.markets[market]
        try:
            # Let an OCA sibling fill (timed exit) that cancelled the stop reconcile first.
            await asyncio.sleep(self.flatten_delay)
            if not s.qty:
                self.store.event('FLATTEN_NOT_NEEDED',dict(market=market,reason=reason));return
            confirmed = True
            for oid in [s.stop_order,s.time_order]:
                if not oid:
                    continue
                if self.broker.status(oid) == 'Filled':
                    confirmed = False;continue
                self.cancel_requested.add(oid)
                try:self.broker.cancel(oid)
                except Exception as exc:self.store.event('CANCEL_FAILED',dict(id=oid,error=str(exc)))
                confirmed = await self._await_cancelled(oid,self.cancel_timeout) and confirmed
            if not s.qty:
                self.store.event('FLATTEN_NOT_NEEDED',dict(market=market,reason=reason));return
            if not confirmed:
                self.halt(f'EMERGENCY_FLATTEN_ABORTED_UNCONFIRMED_CANCEL:{market}')
                self.loud(f'{market} EMERGENCY FLATTEN NOT SENT: exit-order cancels not confirmed by TWS; FLATTEN BY HAND IN TWS')
                return
            snapshot = await asyncio.wait_for(self.broker.snapshot(),5.)
            held = snapshot['positions'].get(m.execution.con_id,0)
            if held != s.side*s.qty or not s.qty:
                self.store.event('FLATTEN_SKIPPED',dict(market=market,broker=held,journal=s.side*s.qty))
                self.halt(f'EMERGENCY_FLATTEN_SKIPPED_POSITION_MISMATCH:{market}')
                self.loud(f'{market} EMERGENCY FLATTEN SKIPPED: broker position {held} != journal {s.side*s.qty}; FLATTEN BY HAND IN TWS')
                return
            qty = s.qty
            s.phase = 'FLATTENING';self.store.save(s)
            oid = self._send(s,'FLATTEN',dict(kind='MKT',side=-s.side,qty=qty,tif='DAY'))
            self.store.event('EMERGENCY_FLATTEN',dict(market=market,id=oid,qty=qty,reason=reason))
            self.loud(f'{market} EMERGENCY FLATTEN SENT: MKT {"SELL" if s.side==1 else "BUY"} {qty} {m.execution.symbol} (order {oid}) after {reason}')
        except Exception as exc:
            self.halt(f'EMERGENCY_FLATTEN_FAILED:{market}:{exc}')
            self.loud(f'{market} EMERGENCY FLATTEN FAILED ({exc}); FLATTEN BY HAND IN TWS')

    def _send(self, state, role, body, order_id=None):
        if order_id is None:
            order_id = self.broker.next_id()
            body = {**body,'account':self.config.account,'ref':self._ref(state,role)}
            self.store.order(order_id,state.market,role,body)
        else:
            # Updates are journaled separately; original intent remains immutable.
            self.store.event('MODIFY_INTENT',dict(id=order_id,body=body))
        self.broker.send(order_id,self.markets[state.market],body)
        return order_id

    def tick(self, market, timestamp, price):
        s = self.states[market]
        if s.phase=='SKIPPED':
            return
        now = self.clock()
        before=(s.opening,s.phase)
        if s.last_timestamp and (aware(timestamp)-aware(s.last_timestamp)).total_seconds()>self.config.watchdog_stale_seconds and aware(timestamp).astimezone(NY).time()<time(11,30):
            self.halt(f'SIGNAL_STREAM_GAP:{market}')
        side = on_price(s,timestamp,price,self.markets[market].signal.tick,now,
                        stale_seconds=self.config.stale_seconds,open_delay=self.config.max_open_delay_seconds)
        if side is not None or before!=(s.opening,s.phase):
            self.store.save(s)
        if side is None or self.halted:
            return
        s.phase='RESERVING'
        self.store.save(s)
        task = asyncio.create_task(self.enter(s,side))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def fresh_quote(self, market):
        bid,ask,stamp = self.broker.quote(market)
        age = (aware(self.clock())-aware(stamp)).total_seconds()
        if not -1<=age<=self.config.stale_seconds:
            raise ValueError('Stale execution quote')
        s = self.states[market]
        if not s.last_timestamp or (self.clock()-aware(s.last_timestamp)).total_seconds()>self.config.stale_seconds:
            raise ValueError('Stale signal trade')
        # A large mini/micro dislocation is not an acceptable executable substitute.
        if abs((bid+ask)/2-s.previous)>max(s.distance,.002*s.previous):
            raise ValueError('Signal/execution price dislocation')
        return bid,ask

    def _skip(self, s, exc):
        # A stale/dislocated quote skips this signal only; no attempt is consumed.
        self.store.event('SKIP_SIGNAL',dict(market=s.market,reason=str(exc)))
        s.phase='FLAT';self.store.save(s)

    async def _send_time_exit(self, s):
        """Broker-held 15:55 exit for an acknowledged stop. Sent even when halted: it can only reduce."""
        if s.time_order or not s.qty or s.market in self.flattened:
            return
        s.time_order = self._send(s,'TIME',dict(kind='MKT',side=-s.side,qty=s.qty,tif='GTC',
            oca=s.oca,good_after=f'{s.day.replace("-","")} 15:55:00 America/New_York'))
        self.store.save(s)
        try:
            await self.broker.wait_ack(s.time_order,3.)
        except Exception:
            # An OCA cancel after the stop already closed the position is expected.
            if s.qty:
                raise

    async def enter(self,s,side):
        try:
            async with self.entry_lock:
                if self.halted:
                    s.phase='FLAT';self.store.save(s);return
                m = self.markets[s.market]
                equity = await asyncio.wait_for(self.broker.equity(),10.)
                try:bid,ask = self.fresh_quote(s.market)
                except (KeyError,ValueError) as exc:
                    self._skip(s,exc);return
                plan = size_order(s,m,side,bid,ask,equity,self.config)
                if plan['qty']<1:
                    s.phase='FLAT';self.store.save(s);return
                if self.config.mode!='live':
                    # Live pilot margin is checked at connect and at the 09:25 preflight.
                    await asyncio.wait_for(self.broker.check_margin(m,plan,equity),10.)
                # Revalidate after asynchronous account/margin checks. Never chase a stale signal.
                try:self.fresh_quote(s.market)
                except (KeyError,ValueError) as exc:
                    self._skip(s,exc);return
                if self.halted or self.clock().astimezone(NY).time()>=time(11,30):
                    raise ValueError('Entry window closed or service halted')
                daily = self.store.get('daily_reserved',0.)
                opened = sum(x.planned_risk for x in self.states.values())
                if daily+plan['risk']>equity*self.config.max_daily_risk_bps/10000 or opened+plan['risk']>equity*self.config.max_open_risk_bps/10000:
                    self.store.event('SKIP_RISK_CAP',{'market':s.market});s.phase='FLAT';self.store.save(s);return
                # Reserve risk and attempt before invoking transport. Conservative daily
                # budget is retained even for a zero-fill IOC or an uncertain outcome.
                oid = self.broker.next_id()
                with self.store.transaction():
                    s.phase='PENDING';s.side=side;s.attempts+=1
                    body = dict(kind='LMT',side=side,qty=plan['qty'],limit=plan['limit'],tif='IOC',
                                account=self.config.account,ref=self._ref(s,'ENTRY'))
                    s.long_armed=s.short_armed=False
                    s.entry_order=oid;s.stop_order=s.time_order=0
                    s.oca=f'OB-{self.config.client_id}-{oid}'
                    s.planned_risk=plan['risk']
                    self.store.set('daily_reserved',daily+plan['risk'])
                    self.store.save(s)
                    self.store.order(oid,s.market,'ENTRY',body)
                self.broker.send(oid,m,body)
                await self.broker.wait_terminal(oid,5.)
                if s.qty:
                    # No early return when halted: an acknowledged stop always gets its timed exit.
                    await self.broker.wait_ack(s.stop_order,3.)
                    if not s.qty:
                        s.phase='CLOSING';self.store.save(s);return
                    if s.market in self.flattened:
                        return
                    await self._send_time_exit(s)
                    if s.market in self.flattened:
                        return
                    s.phase='OPEN' if s.qty else 'CLOSING'
                else:
                    s.phase='CLOSING' if s.stop_order else 'FLAT'
                    if not s.stop_order:s.planned_risk=0.
                self.store.save(s)
        except Exception as exc:
            self.halt(f'ENTRY_OR_PROTECTION:{s.market}:{exc}')
            self.store.save(s)

    async def _late_time_exit(self, s):
        try:
            await self.broker.wait_ack(s.stop_order,3.)
            await self._send_time_exit(s)
        except Exception as exc:
            self.halt(f'LATE_ENTRY_PROTECTION:{s.market}:{exc}')
            self.loud(f'{s.market} LATE ENTRY PROTECTION INCOMPLETE ({exc}); CHECK TWS')

    def _cancel_siblings(self, s, filled_oid):
        """After an exit closes the position, cancel remaining exits explicitly instead of trusting OCA."""
        pending = []
        for oid in [s.stop_order,s.time_order]:
            if oid and oid != filled_oid and self.broker.status(oid) not in TERMINAL:
                self.cancel_requested.add(oid)
                try:self.broker.cancel(oid)
                except Exception as exc:self.store.event('CANCEL_FAILED',dict(id=oid,error=str(exc)))
                if self.broker.status(oid) not in TERMINAL:
                    pending.append(oid)
        if pending:
            self._spawn(self._check_siblings(s.market,pending))

    async def _check_siblings(self, market, ids):
        await asyncio.sleep(self.sibling_timeout)
        left = [oid for oid in ids if self.broker.status(oid) not in TERMINAL]
        if left:
            self.halt(f'ORPHANED_EXIT_ORDER:{market}:{left}')
            self.loud(f'{market} ORPHANED EXIT ORDER(S) {left} STILL WORKING AFTER FLAT; CANCEL BY HAND IN TWS')

    def fill(self,order_id,exec_id,qty,price):
        """Called synchronously by transport for every actual execution, not status fills."""
        order = self.store.orders().get(order_id)
        if order is None:
            self.halt(f'UNKNOWN_EXECUTION:{order_id}');return
        s = self.states[order['market']]
        if not math.isfinite(qty) or qty<=0 or int(qty)!=qty or not math.isfinite(price) or price<=0:
            self.halt('INVALID_EXECUTION');return
        qty=int(qty)
        late=None
        try:
            with self.store.transaction():
                if not self.store.add_fill(exec_id,dict(order_id=order_id,qty=qty,price=price)):
                    return
                if order['role']=='ENTRY':
                    if order_id!=s.entry_order or s.phase not in {'PENDING','OPEN'}:
                        # A real execution after the IOC looked terminal: record and protect it, then halt.
                        if order_id!=s.entry_order:
                            if s.qty:
                                raise ValueError(f'Late entry execution {order_id} while another position is open')
                            s.entry_order=order_id;s.side=order['body']['side'];s.entry_value=0.
                            s.stop_order=s.time_order=0;s.oca=f'OB-{self.config.client_id}-{order_id}'
                        late=f'LATE_ENTRY_EXECUTION:{s.market}:{order_id}'
                        s.phase='OPEN'
                    s.entry_value+=qty*price;s.qty+=qty
                    s.stop=snap(s.entry-s.side*s.distance,self.markets[s.market].execution.tick,s.side==-1)
                else:
                    if qty>s.qty:
                        raise ValueError('Exit exceeds journal position')
                    avg=s.entry;s.qty-=qty;s.entry_value=avg*s.qty
                    if s.qty==0:s.phase='CLOSING'
                self.store.save(s)
            if order['role']=='ENTRY':
                # Protect each partial fill immediately; do not wait for parent completion.
                body=dict(kind='STP',side=-s.side,qty=s.qty,stop=s.stop,tif='GTC',oca=s.oca,
                          account=self.config.account,ref=self._ref(s,'STOP'))
                s.stop_order=self._send(s,'STOP',body,s.stop_order or None)
                self.store.save(s)
                self.notify(f'{s.market} ENTRY FILL ({self.config.mode}): {"BUY" if s.side==1 else "SELL"} {qty} '
                            f'{self.markets[s.market].execution.symbol} @ {price}; stop {s.stop} sent (order {s.stop_order})')
                if late:
                    self.halt(late)
                    self.loud(f'{s.market} LATE ENTRY FILL recorded and stop sent; session halted for review')
                    self._spawn(self._late_time_exit(s))
            elif s.qty==0:
                self._cancel_siblings(s,order_id)
        except Exception as exc:
            self.halt(f'EXECUTION_OR_STOP:{exc}')

    def _expected(self):
        return {m.execution.con_id:self.states[m.name].side*self.states[m.name].qty for m in self.config.markets}

    async def watchdog(self):
        """No automatic resubmission on uncertainty; retain resting protection."""
        try:
            before = self._expected()
            snapshot = await asyncio.wait_for(self.broker.snapshot(),5.)
            orders=self.store.orders()
            expected = self._expected()
            # Include client identity: API order IDs are only unique within a client.
            for o in snapshot['orders']:
                if o['con_id'] in expected and (o['client_id']!=self.config.client_id or o['id'] not in orders or o['ref']!=orders[o['id']]['body']['ref']):
                    raise ValueError('Unowned order on configured execution contract')
            # A fill in flight during the snapshot await makes one comparison meaningless;
            # only a discrepancy on consecutive stable checks halts.
            stable = before == expected
            issues = []
            if stable:
                issues += ['Broker/journal position mismatch' for cid,qty in expected.items() if snapshot['positions'].get(cid,0)!=qty]
            working={o['id']:o for o in snapshot['orders'] if o['client_id']==self.config.client_id}
            late = self.clock().astimezone(NY).time()>=time(15,56)
            for s in self.states.values():
                if s.phase=='SKIPPED' and not s.qty:
                    continue
                local=self.clock().astimezone(NY)
                boundary=local.replace(hour=9,minute=30,second=0,microsecond=0)
                if not s.opening and (local-boundary).total_seconds()>self.config.max_open_delay_seconds:
                    if s.phase!='BLOCKED':
                        s.phase='BLOCKED';s.note='MISSED_0930_OPEN';self.store.save(s)
                        self.store.event('MARKET_BLOCKED',dict(market=s.market,reason=s.note))
                    continue
                if stable and s.qty and s.phase=='OPEN':
                    stop=working.get(s.stop_order)
                    if not stop or stop['remaining']!=s.qty or stop['kind']!='STP' or stop['side']!=-s.side or stop['stop']!=s.stop:
                        issues.append('Broker protective stop differs from journal')
                if s.opening and local.time()<time(11,30) and (self.clock()-aware(s.last_timestamp)).total_seconds()>self.config.watchdog_stale_seconds:
                    raise ValueError('Signal stream stale')
                if s.phase=='CLOSING':
                    ids=[s.entry_order,s.stop_order,s.time_order]
                    if all(not oid or self.broker.status(oid) in TERMINAL for oid in ids):
                        s.phase='FLAT';s.planned_risk=0.;self.store.save(s)
                if s.qty and s.phase=='OPEN' and (not s.stop_order or self.broker.status(s.stop_order) not in ACKNOWLEDGED):
                    raise ValueError('Position lacks acknowledged protective stop')
                if late and s.qty:
                    raise ValueError('Timed exit not complete: inspect broker immediately')
            if late:
                strays=[o for o in snapshot['orders'] if o['con_id'] in expected and STRATEGY_REF in str(o.get('ref') or '')]
                if strays:
                    raise ValueError('OpenBreakout order still working after 15:56: cancel in TWS')
            if issues:
                self.mismatches += 1
                self.store.event('RECONCILE_DISCREPANCY',dict(count=self.mismatches,issues=issues))
                if self.mismatches >= MISMATCH_CHECKS:
                    raise ValueError(issues[0])
            elif stable:
                self.mismatches = 0
            for state in self.states.values():self.store.save(state)
            self.store.set('heartbeat',dict(at=self.clock().isoformat(),halted=self.halted,positions=expected))
        except Exception as exc:
            self.halt(f'RECONCILE:{exc}')

    async def drain(self):
        while self.tasks: await asyncio.gather(*list(self.tasks))
