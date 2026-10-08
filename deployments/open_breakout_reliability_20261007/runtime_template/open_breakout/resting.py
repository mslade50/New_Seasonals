"""Broker-held stop-limit entry lifecycle; the IOC path remains replayable."""
import asyncio
import math
from datetime import time

from .strategy import NY, snap, size_order


class RestingEntries:
    @property
    def resting_entries(self):
        return self.config.entry_order_type == 'stop_limit'

    def working_entries(self, s):
        return [oid for oid in s.entry_orders
                if self.broker.status(oid) not in {'Filled','Cancelled','ApiCancelled','Inactive','REJECTED'}]

    def queue_resting(self, s):
        if (not self.resting_entries or self.halted or s.market in self.manual_markets
                or s.phase != 'FLAT' or not s.opening or s.attempts >= 3
                or self.clock().astimezone(NY).time() >= time(11,30)):
            return
        s.phase = 'RESERVING'
        self.store.save(s)
        self._spawn(self.park_entries(s))

    async def park_entries(self, s):
        try:
            async with self.entry_lock:
                if self.halted or s.market in self.manual_markets:
                    s.phase='FLAT';self.store.save(s);return
                m=self.markets[s.market]
                equity=await asyncio.wait_for(self.broker.equity(),10.)
                bid,ask=self.fresh_quote(s.market)
                if self.clock().astimezone(NY).time() >= time(11,30):
                    s.phase='FLAT';self.store.save(s);return
                daily=self.store.get('daily_reserved',0.)
                opened=sum(x.planned_risk for x in self.states.values())
                room=min(equity*self.config.max_daily_risk_bps/10000-daily,
                         equity*self.config.max_open_risk_bps/10000-opened)
                plans=[]
                for side,armed in ((1,s.long_armed),(-1,s.short_armed and s.score>=20)):
                    trigger=snap(s.opening+side*s.distance,m.execution.tick,side==1)
                    # Only place an untriggered stop. After an exit, a fresh return
                    # inside the boundary is required on both mini and micro prices.
                    inside=(max(s.previous,ask)<trigger if side==1 else min(s.previous,bid)>trigger)
                    if not armed or not inside:
                        continue
                    plan=size_order(s,m,side,trigger,trigger,equity,self.config)
                    qty=min(plan['qty'],max(0,math.floor(room/plan['per_contract']+1e-9)),
                            self.margin_day_cap.get(s.market,plan['qty']))
                    if qty:
                        plans.append({**plan,'qty':qty,'risk':qty*plan['per_contract'],'trigger':trigger})
                if not plans:
                    s.phase='FLAT';self.store.save(s);return
                if self.config.mode!='live':
                    for plan in plans:
                        await asyncio.wait_for(self.broker.check_margin(m,plan,equity),10.)
                bid,ask=self.fresh_quote(s.market)
                if self.halted or s.market in self.manual_markets or self.clock().astimezone(NY).time()>=time(11,30):
                    s.phase='FLAT';self.store.save(s);return
                plans=[p for p in plans if (max(s.previous,ask)<p['trigger'] if p['side']==1
                                           else min(s.previous,bid)>p['trigger'])]
                if not plans:
                    s.phase='FLAT';self.store.save(s);return
                risk=max(p['risk'] for p in plans) # OCA with block: one side can fill.
                with self.store.transaction():
                    s.attempts+=1;s.phase='RESTING';s.entry_orders=[];s.entry_risks={}
                    s.entry_order=s.stop_order=s.time_order=0;s.side=0
                    s.planned_risk=risk;s.long_armed=s.short_armed=False
                    self.store.set('daily_reserved',daily+risk)
                    self.store.save(s)
                group=f'OB-ENTRY-{self.config.client_id}-{s.day}-{s.market}-{s.attempts}'
                for plan in plans:
                    if s.qty or self.halted or s.market in self.manual_markets:
                        break
                    oid=self.broker.next_id()
                    side=plan['side'];symbol=m.execution.symbol
                    body=dict(kind='STP LMT',side=side,qty=plan['qty'],stop=plan['trigger'],
                              limit=plan['limit'],tif='GTD',
                              good_till=f'{s.day.replace("-","")} 11:30:00 America/New_York',
                              trigger_method=2,oca=group,oca_type=1,account=self.config.account,
                              ref=f'{symbol}|{"BUY" if side==1 else "SELL"}|OpenBreakout|{s.day}|{s.market}-{s.attempts}-ENTRY')
                    with self.store.transaction():
                        s.entry_orders.append(oid);s.entry_risks[str(oid)]=plan['risk']
                        self.store.order(oid,s.market,'ENTRY',body);self.store.save(s)
                    self.broker.send(oid,m,body)
                for oid in s.entry_orders:
                    if self.broker.status(oid) not in {'Filled','Cancelled','ApiCancelled'}:
                        await self.broker.wait_ack(oid,3.)
                submitted=[self.store.orders()[oid]['body'] for oid in s.entry_orders]
                self.notify(f'{s.market} RESTING ENTRIES: '+', '.join(
                    f'{"BUY" if p["side"]==1 else "SELL"} {p["qty"]} {m.execution.symbol} '
                    f'stop {p["stop"]} limit {p["limit"]}' for p in submitted)+'; expires 11:30 ET')
        except (KeyError,ValueError) as exc:
            if s.entry_orders:
                self.halt(f'RESTING_ENTRY:{s.market}:{exc}')
            else:
                self._skip(s,exc)
        except Exception as exc:
            self.halt(f'RESTING_ENTRY:{s.market}:{exc}')

    async def cancel_resting_entries(self, market=None):
        """Cancel entries only, retaining the stop and timed exit for actual fills."""
        if not self.resting_entries:
            return
        async with self.entry_cancel_lock:
            await self._cancel_resting_entries(market)

    async def _cancel_resting_entries(self, market):
        for s in self.states.values():
            if (market is not None and s.market!=market) or s.market in self.manual_markets:
                continue
            pending=self.working_entries(s)
            for oid in pending:
                self.cancel_requested.add(oid)
                try:self.broker.cancel(oid)
                except Exception as exc:
                    self.store.event('ENTRY_CANCEL_FAILED',dict(id=oid,error=str(exc)))
            for oid in pending:
                if not await self._await_cancelled(oid,self.cancel_timeout) and self.broker.status(oid)!='Filled':
                    self.loud(f'{s.market} ENTRY CANCEL UNCONFIRMED: order {oid}; CHECK TWS')
                    self.halt(f'ENTRY_CANCEL_UNCONFIRMED:{s.market}:{oid}')
            if not s.qty and s.phase=='RESTING' and not self.working_entries(s):
                s.phase='FLAT';s.planned_risk=0.;self.store.save(s)

    async def finish_resting_fill(self, s):
        try:
            # Cancel both the opposite entry and any unfilled parent remainder.
            await self.cancel_resting_entries(s.market)
            if s.qty and s.market not in self.manual_markets:
                await self.broker.wait_ack(s.stop_order,3.)
                await self._send_time_exit(s)
                if s.qty and s.market not in self.flattened:
                    s.phase='OPEN';self.store.save(s)
        except Exception as exc:
            self.halt(f'RESTING_PROTECTION:{s.market}:{exc}')
        finally:
            self.resting_fill_tasks.discard(s.market)

    async def check_entry_cancel(self, s, oid):
        # OCA cancellation/status can arrive before the sibling execution report.
        await asyncio.sleep(.25)
        if s.qty or oid in self.cancel_requested or self.clock().astimezone(NY).time()>=time(11,30):
            return
        for other in s.entry_orders:
            if other!=oid and self.broker.status(other)=='Filled':
                try:await self.broker.wait_terminal(other,3.)
                except Exception:pass
                if s.qty:return
        self.halt(f'RESTING_ENTRY_CANCELLED:{s.market}:{oid}')
