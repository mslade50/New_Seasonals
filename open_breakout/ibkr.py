"""IBKR transport. Importing this module never connects. Live routing requires the live config limits and session ack."""
import asyncio
from datetime import datetime, timezone
import math
from .service import TERMINAL, ACKNOWLEDGED, STRATEGY_REF, ref_strategy, exec_key
from .strategy import NY

EXEC_CACHE_SECONDS = 1.
# Inside the watchdog's 5 s snapshot budget; a slow executions call makes own position UNKNOWN, not a halt.
EXEC_TIMEOUT_SECONDS = 3.
RECOVERY_REQUEST_TIMEOUT_SECONDS = 10.
RECOVERY_STABLE_SECONDS = .25

TRANSPORT_LOSS_CODES = {1100, 1300}
TRANSPORT_RESTORE_CODES = {1101, 1102}
FATAL_ERROR_CODES = {201,10147,10148,354,10167,10168,10089,10090,10189}


def session_start(now=None):
    """Midnight New York of the current day: reqExecutions can return up to seven days (TWS Trade Log setting)."""
    local = (now or datetime.now(timezone.utc)).astimezone(NY)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def fully_executed(trade):
    """Execution reports can confirm a complete fill before orderStatus catches up."""
    if trade.orderStatus.status == 'Filled':
        return True
    return bool(trade.fills and trade.order.totalQuantity > 0
                and trade.fills[-1].execution.cumQty >= trade.order.totalQuantity)


def attribute_executions(fills, account, since=None):
    """Signed position per conId from execution reports: (OpenBreakout's own, other strategies', own exec keys).
    Attribution is the execution's orderRef strategy field; executions without one (manual TWS orders)
    are in neither book. An IB correction reuses the execId stem with a higher suffix and replaces it.
    Executions before `since` (earlier sessions) are ignored."""
    latest = {}
    for f in fills:
        e = f.execution
        if e.acctNumber != account:
            continue
        stamp = getattr(e, 'time', None)
        if since is not None and isinstance(stamp, datetime) and stamp.tzinfo and stamp < since:
            continue
        key = exec_key(e.execId)
        suffix = e.execId.rpartition('.')[2]
        if key not in latest or suffix > latest[key][0]:
            latest[key] = (suffix, f)
    own, other, own_ids = {}, {}, set()
    for key, (_, f) in latest.items():
        e = f.execution
        strategy = ref_strategy(e.orderRef)
        if not strategy:
            continue
        book = own if strategy == STRATEGY_REF else other
        if book is own:
            own_ids.add(key)
        cid = f.contract.conId
        book[cid] = book.get(cid,0)+(e.shares if e.side == 'BOT' else -e.shares)
    return own, other, own_ids


class IBKR:
    def __init__(self,config,session=None):
        from ib_insync import IB
        self.config=config;self.session=session;self.ib=IB();self.contracts={};self.quotes={};self.trades={}
        self.fill_callback=lambda *a:None
        self.status_callback=lambda *a:None
        self.halt_callback=lambda *a:None
        self.order_error_callback=lambda *a:None
        self.diagnostic_callback=lambda *a:None
        # A socket connection or a 1101/1102 message is not health proof.  Every
        # adapter starts in a new, unproved connection epoch and only a complete,
        # stable reconciliation may set healthy=True.
        self.healthy=False;self.ack=None;self.contract_report={}
        self.last_seen={};self.data_types={}
        self.net_liq=self.excess=float('nan')
        # ib_insync keys openOrders/positions requests by a single name; overlapping calls orphan one.
        self.snapshot_lock=asyncio.Lock()
        self._exec_books=None;self._exec_at=0.
        # Bumped by every own execution; a request that overlapped one is never cached.
        self._exec_generation=0
        # execIds already handed to fill_callback (live or re-delivered from reqExecutions).
        self._delivered=set()
        self._connection_epoch=1
        self._epoch_started_at=datetime.now(timezone.utc)
        self._book_revision=0
        self._position_signatures={}
        self._order_signatures={}
        self._account_signatures={}
        self._fatal_reason=None
        self._restore_signal=None
        self._needs_resubscribe=True
        self._events_installed=False
        self._ticker_handler_installed=False
        self._subscription_epoch=None
        self._on_tick=lambda *a:None
        self._capture=lambda *a:None
        self._retired=False

    @property
    def connection_epoch(self):return self._connection_epoch

    def _install_events(self):
        """Install each SDK handler once, before connectAsync can emit anything."""
        if self._events_installed:return
        self.ib.execDetailsEvent+=self._fill
        self.ib.orderStatusEvent+=self._status
        # These callbacks make an in-flight recovery snapshot unstable.  They
        # deliberately exclude market ticks, which have their own freshness gate.
        if hasattr(self.ib,'openOrderEvent'):self.ib.openOrderEvent+=self._open_order_changed
        if hasattr(self.ib,'positionEvent'):self.ib.positionEvent+=self._position_changed
        if hasattr(self.ib,'accountValueEvent'):self.ib.accountValueEvent+=self._account_value_changed
        self.ib.disconnectedEvent+=self._disconnected
        self.ib.errorEvent+=self._error
        self._events_installed=True

    def _book_changed(self,kind):
        if self._retired:return
        self._book_revision+=1

    def _relevant_contract(self,con_id):
        return con_id in {m.execution.con_id for m in self.config.markets}

    def _position_signature(self,position):
        if (position.account!=self.config.account
                or not self._relevant_contract(position.contract.conId)):return None
        return (position.account,position.contract.conId),float(position.position)

    def _position_changed(self,position):
        if self._retired:return
        item=self._position_signature(position)
        if item is None:return
        key,value=item
        if self._position_signatures.get(key)==value:return
        self._position_signatures[key]=value;self._book_changed('POSITION')

    def _order_signature(self,trade):
        order=trade.order;contract=trade.contract;status=trade.orderStatus
        if (order.account!=self.config.account
                or not self._relevant_contract(contract.conId)):return None
        key=(order.clientId,order.orderId,getattr(order,'permId',0))
        value=(status.status,float(status.filled),float(order.totalQuantity),order.orderType,
               order.action,float(getattr(order,'auxPrice',0.) or 0.),
               float(getattr(order,'lmtPrice',0.) or 0.),order.orderRef)
        return key,value

    def _open_order_changed(self,trade):
        if self._retired:return
        item=self._order_signature(trade)
        if item is None:return
        key,value=item
        if self._order_signatures.get(key)==value:return
        self._order_signatures[key]=value;self._book_changed('OPEN_ORDER')

    def _account_signature(self,value):
        if (value.account!=self.config.account or value.currency!='USD'
                or value.tag not in {'NetLiquidation','ExcessLiquidity'}):return None
        return (value.account,value.tag,value.currency),float(value.value)

    def _account_value_changed(self,value):
        if self._retired:return
        item=self._account_signature(value)
        if item is None:return
        key,signature=item
        if self._account_signatures.get(key)==signature:return
        self._account_signatures[key]=signature;self._book_changed('ACCOUNT_VALUE')

    def _seed_recovery_signatures(self,positions,trades,values):
        """Baseline callback comparisons from the exact proved snapshot."""
        self._position_signatures={k:v for p in positions
                                   for item in [self._position_signature(p)] if item is not None
                                   for k,v in [item]}
        self._order_signatures={k:v for t in trades
                                for item in [self._order_signature(t)] if item is not None
                                for k,v in [item]}
        self._account_signatures={(self.config.account,tag,'USD'):float(value)
                                  for tag,value in values.items()}

    def _invalidate_recovery(self,reason,*,restore_code=None,notify=True):
        """Start a new evidence epoch.  No prior callback can prove this epoch healthy."""
        if self._retired:return
        self._connection_epoch+=1
        self._epoch_started_at=datetime.now(timezone.utc)
        self._book_revision+=1
        self.healthy=False
        self._position_signatures.clear();self._order_signatures.clear();self._account_signatures.clear()
        self._exec_books=None;self._exec_at=0.
        self.last_seen.clear();self.data_types.clear()
        self._restore_signal=restore_code
        if restore_code==1101:self._needs_resubscribe=True
        self.diagnostic_callback(dict(kind='transport_epoch',epoch=self._connection_epoch,
                                      reason=str(reason),restore_code=restore_code,
                                      at=self._epoch_started_at.isoformat()))
        if notify:self.halt_callback(reason)

    def _disconnected(self):
        self._restore_signal=None
        self._needs_resubscribe=True
        self._invalidate_recovery('IB_DISCONNECTED')

    async def connect(self):
        # Live: structural pilot limits plus the exact session/account acknowledgement.
        self.ack=self.config.authorize(self.session)
        self._install_events()
        await self.ib.connectAsync(self.config.host,self.config.port,clientId=self.config.client_id,
                                   readonly=self.config.mode=='shadow',account=self.config.account,timeout=10)
        if self.config.account not in self.ib.managedAccounts():
            self.ib.disconnect();raise ValueError('Configured account is not available on this connection')
        self.ib.reqMarketDataType(1)
        from ib_insync import Contract
        for m in self.config.markets:
            for spec in [m.signal,m.execution]:
                if spec.con_id in self.contracts:continue
                details=await self.ib.reqContractDetailsAsync(Contract(conId=spec.con_id,exchange='CME'))
                if len(details)!=1:raise ValueError('Ambiguous futures contract')
                d=details[0];c=d.contract
                if c.secType!='FUT' or c.symbol!=spec.symbol or c.lastTradeDateOrContractMonth[:8]!=spec.expiry or float(c.multiplier)!=spec.multiplier or d.minTick!=spec.tick or c.currency!='USD' or c.exchange!='CME':
                    raise ValueError('Contract identity, expiry, multiplier or tick mismatch')
                self.contracts[spec.con_id]=c
                self.contract_report[spec.symbol]=dict(con_id=c.conId,symbol=c.symbol,local_symbol=c.localSymbol,
                    expiry=c.lastTradeDateOrContractMonth,multiplier=float(c.multiplier),min_tick=d.minTick,exchange=c.exchange,ok=True)

    def _halt(self,reason):
        self._fatal_reason=str(reason)
        self._invalidate_recovery(reason,notify=True)

    def _error(self,req_id,code,message,contract):
        if self._retired:return
        self.diagnostic_callback(dict(kind='ib_error',req_id=req_id,code=code,message=message,
                                      epoch=self._connection_epoch))
        if req_id in self.trades:
            self.order_error_callback(req_id,code,message)
            # A cancel can lose the race with the final fill. Keep the error in
            # the journal, but a verified complete fill is not a transport failure.
            if code == 10148 and fully_executed(self.trades[req_id]):
                return
        reason=f'IB_ERROR:{code}:{message}'
        if code in TRANSPORT_LOSS_CODES:
            self._restore_signal=None;self._needs_resubscribe=True
            self._invalidate_recovery(reason)
            return
        if code in TRANSPORT_RESTORE_CODES:
            # Coalesce duplicate restore messages for the same epoch.  A restore
            # message begins a fresh proof epoch; it never clears health itself.
            # Once that epoch has been proved healthy, even the same code is a
            # new restoration boundary (the preceding loss callback may have
            # been missed).  Post-arm it must halt new entries immediately.
            if self._restore_signal==code and not self.healthy:return
            was_healthy=self.healthy
            self._invalidate_recovery(reason,restore_code=code,notify=was_healthy)
            if code==1101:self._request_subscriptions()
            return
        # 2103/2105 farm blips are warnings: stream staleness catches a real data outage.
        if code in FATAL_ERROR_CODES:self._halt(reason)

    def _fill(self,trade,fill):
        e=fill.execution
        if e.acctNumber==self.config.account and e.clientId==self.config.client_id:
            # A new own execution: the next own-position read must not use the cached books.
            self._exec_books=None;self._exec_generation+=1;self._book_revision+=1
            self._delivered.add(e.execId)
            self.fill_callback(e.orderId,e.execId,e.shares,e.price)

    def _status(self,trade):
        if trade.order.account==self.config.account and trade.order.clientId==self.config.client_id:
            self._book_revision+=1
            self.status_callback(trade.order.orderId,trade.orderStatus.status)

    def next_id(self):return self.ib.client.getReqId()

    def _order(self,oid,body):
        from ib_insync import Order
        fields=dict(orderId=oid,clientId=self.config.client_id,account=self.config.account,
                    action='BUY' if body['side']==1 else 'SELL',totalQuantity=body['qty'],
                    orderType=body['kind'],tif=body['tif'],orderRef=body['ref'],transmit=True,outsideRth=True)
        if 'limit' in body:fields['lmtPrice']=body['limit']
        if 'stop' in body:fields['auxPrice']=body['stop']
        if 'oca' in body:fields.update(ocaGroup=body['oca'],ocaType=body.get('oca_type',2))
        if 'good_after' in body:fields['goodAfterTime']=body['good_after']
        if 'good_till' in body:fields['goodTillDate']=body['good_till']
        if 'trigger_method' in body:fields['triggerMethod']=body['trigger_method']
        return Order(**fields)

    def send(self,oid,market,body):
        self.config.authorize(self.session)
        if self.config.mode not in {'paper','live'}:
            raise PermissionError('Order transmission is available only in paper or authorized live mode')
        if body.get('account')!=self.config.account:
            raise PermissionError('Order account mismatch')
        if self.config.mode=='live':
            qty,cap=body['qty'],self.config.max_contracts_for(market)
            if isinstance(qty,bool) or not isinstance(qty,(int,float)) or not math.isfinite(qty) or qty!=int(qty) or not 0<qty<=cap:
                raise PermissionError(f'Live quantity cap exceeded: qty {qty} must be a whole number in [1, {cap}]')
        # A halted feed still permits protective exits for actual executions.
        if body['kind'] in {'LMT','STP LMT'} and (not self.healthy or not self.ib.isConnected()):
            raise RuntimeError('Unhealthy broker connection')
        order=self._order(oid,body)
        self.trades[oid]=self.ib.placeOrder(self.contracts[market.execution.con_id],order)

    def cancel(self,oid):
        self.config.authorize(self.session)
        if self.config.mode not in {'paper','live'}:
            raise PermissionError('Order cancellation is available only in paper or authorized live mode')
        trade=self.trades.get(oid)
        if trade is None:raise ValueError(f'Unknown order {oid}')
        if fully_executed(trade):
            return
        self.ib.cancelOrder(trade.order)

    def status(self,oid):
        trade=self.trades.get(oid)
        return trade.orderStatus.status if trade else 'UNKNOWN'

    def last_error_code(self,oid):
        trade=self.trades.get(oid)
        if not trade or not trade.log:return None
        return trade.log[-1].errorCode or None

    async def _wait(self,oid,predicate,timeout):
        deadline=asyncio.get_running_loop().time()+timeout
        while asyncio.get_running_loop().time()<deadline:
            status=self.status(oid)
            if predicate(status):return status
            if status in {'Inactive','REJECTED'}:raise RuntimeError(f'Order {oid} rejected')
            await asyncio.sleep(.05)
        raise TimeoutError(f'Uncertain acknowledgement/order state: {oid}')

    async def wait_terminal(self,oid,timeout):
        await self._wait(oid,lambda s:s in TERMINAL,timeout)
        deadline=asyncio.get_running_loop().time()+timeout
        while asyncio.get_running_loop().time()<deadline:
            trade=self.trades[oid]
            if sum(f.execution.shares for f in trade.fills)==trade.orderStatus.filled:return
            await asyncio.sleep(.05)
        raise TimeoutError('Terminal status has unreconciled executions')
    async def wait_ack(self,oid,timeout):return await self._wait(oid,lambda s:s in ACKNOWLEDGED,timeout)

    def quote(self,name):
        if not self.healthy:raise ValueError('Broker data feed is unhealthy')
        return self.quotes[name]

    async def account_values(self,force=False):
        if force:
            # ib_insync.accountSummaryAsync() returns its persistent cache when
            # populated, which cannot prove recovery after an outage.  Clear
            # that cache and require a new reqAccountSummary completion.  The
            # recovery path never fills missing tags from the older account-
            # values stream: both required USD tags must be in this response.
            wrapper=getattr(self.ib,'wrapper',None)
            cache=getattr(wrapper,'acctSummary',None)
            request=getattr(self.ib,'reqAccountSummaryAsync',None)
            if cache is None or request is None:
                raise RuntimeError('Fresh account summary is unsupported')
            cache.clear()
            await request()
            rows=list(cache.values())
        else:
            rows=await self.ib.accountSummaryAsync()
        values={x.tag:float(x.value) for x in rows if x.account==self.config.account and x.currency=='USD' and x.tag in {'NetLiquidation','ExcessLiquidity'}}
        # connectAsync(account=...) already streams account updates; re-requesting them never
        # completes (verified 2026-09-24 on port 7496), so only fill gaps from that stream.
        if not force:
            for x in self.ib.accountValues(self.config.account):
                if x.currency=='USD' and x.tag in {'NetLiquidation','ExcessLiquidity'} and x.tag not in values:
                    values[x.tag]=float(x.value)
        if any(not math.isfinite(values.get(k,float('nan'))) or values[k]<=0 for k in ['NetLiquidation','ExcessLiquidity']):
            raise ValueError('Positive USD equity and excess liquidity required')
        self.excess=values['ExcessLiquidity'];self.net_liq=values['NetLiquidation']
        return values

    async def equity(self):
        """Sizing basis. Live sizes off the configured capital base; margin uses actual IB values."""
        values=await self.account_values()
        return self.config.shadow_equity if self.config.mode=='live' else values['NetLiquidation']

    def margin_limit(self,equity):
        if self.config.mode=='live':
            return min(self.excess,self.net_liq)*self.config.max_margin_fraction
        return min(self.excess,equity*self.config.max_margin_fraction)

    async def what_if(self,market,body):
        result=await asyncio.wait_for(self.ib.whatIfOrderAsync(self.contracts[market.execution.con_id],self._order(0,body)),10)
        if not result or not hasattr(result,'initMarginChange'):
            raise ValueError('Margin preview unavailable')
        return result

    async def check_margin(self,market,plan,equity):
        body=dict(kind='LMT',side=plan['side'],qty=plan['qty'],limit=plan['limit'],tif='IOC',ref='OB:WHATIF')
        result=await self.what_if(market,body)
        change=float(result.initMarginChange)
        if not math.isfinite(change) or change<0 or change>self.margin_limit(equity) or result.warningText:
            raise ValueError('Margin preview unavailable, warned, or exceeds limit')

    async def _execution_books(self,force=False):
        """Today's account executions split by orderRef; at most one request per second. Hold snapshot_lock.
        No clientId filter: executions from an earlier process, or seen from a read-only client, still count."""
        now=asyncio.get_running_loop().time()
        if force or self._exec_books is None or now-self._exec_at>=EXEC_CACHE_SECONDS:
            from ib_insync import ExecutionFilter
            generation=self._exec_generation
            fills=await asyncio.wait_for(self.ib.reqExecutionsAsync(ExecutionFilter(acctCode=self.config.account)),
                                         EXEC_TIMEOUT_SECONDS)
            since=session_start()
            # ib_insync stores every execId a reqExecutions answer carries and then suppresses the LIVE
            # execDetailsEvent for it. If the answer races ahead of the live report, hand our own
            # execution to the journal here (Service.fill is idempotent on execId).
            for f in fills:
                e=f.execution
                stamp=getattr(e,'time',None)
                if (e.acctNumber==self.config.account and e.clientId==self.config.client_id
                        and e.execId not in self._delivered and not (isinstance(stamp,datetime) and stamp.tzinfo and stamp<since)):
                    self._fill(None,f)
            books=attribute_executions(fills,self.config.account,since)
            if generation!=self._exec_generation:
                # An own fill landed while the request was in flight: use it once, never cache it.
                return books
            self._exec_books=books;self._exec_at=now
        return self._exec_books

    async def own_position(self,con_id):
        """OpenBreakout's signed position in a contract from its own executions (orderRef), not reqPositions.
        Raises when the executions request fails or times out: the caller treats that as UNKNOWN."""
        async with self.snapshot_lock:
            own,_,_=await self._execution_books()
        return own.get(con_id,0)

    async def snapshot(self):
        async with self.snapshot_lock:
            trades=await self.ib.reqAllOpenOrdersAsync()
            positions=await self.ib.reqPositionsAsync()
            try:
                own,other,ids=await self._execution_books()
                own,other,ids,error=dict(own),dict(other),sorted(ids),None
            except Exception as exc:
                # UNKNOWN own position (None): the service skips the position checks and alerts, never halts on it.
                own=other=ids=None;error=f'{type(exc).__name__}: {exc}'
        # positions: whole account (informational and the ceiling); own/foreign: attributed executions.
        return dict(positions={p.contract.conId:p.position for p in positions if p.account==self.config.account},
                    own=own,foreign=other,own_exec_ids=ids,executions_error=error,
                    orders=[dict(id=t.order.orderId,client_id=t.order.clientId,con_id=t.contract.conId,ref=t.order.orderRef,
                        remaining=t.order.totalQuantity-t.orderStatus.filled,kind=t.order.orderType,
                        side=1 if t.order.action=='BUY' else -1,stop=t.order.auxPrice,
                        limit=t.order.lmtPrice,tif=t.order.tif,good_till=t.order.goodTillDate,
                        oca=t.order.ocaGroup,oca_type=t.order.ocaType)
                            for t in trades if t.order.account==self.config.account])

    def recovery_token_valid(self,token):
        """A proof token remains valid only while its exact connection/book epoch is unchanged."""
        needed={f'{m.name}:{k}' for m in self.config.markets for k in ('trade','quote')}
        expected_contracts={s.con_id for m in self.config.markets for s in (m.signal,m.execution)}
        now=datetime.now(timezone.utc)
        streams=(needed<=set(self.last_seen)
                 and all(self.last_seen[k]>=self._epoch_started_at
                         and (now-self.last_seen[k]).total_seconds()<=15. for k in needed)
                 and all(self.data_types.get(cid)==1 for cid in expected_contracts))
        return bool(token and tuple(token)==(self._connection_epoch,self._book_revision)
                    and streams and self.healthy and not self._fatal_reason and self.ib.isConnected())

    async def reconcile_recovery(self,*,wait_seconds=30.,max_age=15.,
                                 request_timeout=RECOVERY_REQUEST_TIMEOUT_SECONDS,
                                 stable_seconds=RECOVERY_STABLE_SECONDS):
        """Prove a connection epoch from fresh, complete, callback-stable broker reads.

        1101/1102 and fresh prices are inputs to this barrier, never recovery by
        themselves.  Any epoch or owner-book callback change while requests are in
        flight invalidates the whole proof.  Missing/timeout responses remain
        unknown; they are never converted to empty positions or orders.
        """
        epoch=self._connection_epoch
        started=self._epoch_started_at
        proof=dict(epoch=epoch,started_at=started.isoformat(),complete=False,retryable=True,
                   restoration_code=self._restore_signal,
                   failures=[],token=None)
        failures=proof['failures']
        if self._fatal_reason:
            failures.append(f'FATAL:{self._fatal_reason}');proof['retryable']=False;return proof
        expected_client=self.config.client_id
        if not self.ib.isConnected():
            failures.append('DISCONNECTED_BEFORE_RECOVERY');return proof
        actual_client=getattr(getattr(self.ib,'client',None),'clientId',None)
        managed=list(self.ib.managedAccounts())
        if actual_client!=expected_client or self.config.account not in managed or not self.session:
            failures.append('IDENTITY_MISMATCH')
            proof.update(retryable=False,expected_client_id=expected_client,actual_client_id=actual_client,
                         account_available=self.config.account in managed,session=self.session)
            return proof
        needed={f'{m.name}:{k}' for m in self.config.markets for k in ('trade','quote')}
        deadline=asyncio.get_running_loop().time()+wait_seconds
        while asyncio.get_running_loop().time()<deadline:
            if epoch!=self._connection_epoch:
                failures.append('EPOCH_CHANGED_DURING_STREAM_WAIT');return proof
            now=datetime.now(timezone.utc)
            if (needed<=set(self.last_seen)
                    and all(self.last_seen[k]>=started and (now-self.last_seen[k]).total_seconds()<=max_age for k in needed)):
                break
            await asyncio.sleep(.05)
        now=datetime.now(timezone.utc)
        missing=sorted(k for k in needed if k not in self.last_seen or self.last_seen[k]<started)
        stale=sorted(k for k in needed if k in self.last_seen and (now-self.last_seen[k]).total_seconds()>max_age)
        if missing:failures.append('STREAMS_MISSING:'+','.join(missing))
        if stale:failures.append('STREAMS_STALE:'+','.join(stale))
        expected_contracts={s.con_id for m in self.config.markets for s in (m.signal,m.execution)}
        bad_types=sorted(cid for cid in expected_contracts if self.data_types.get(cid)!=1)
        if bad_types:failures.append('MARKET_DATA_TYPE_UNPROVED:'+','.join(map(str,bad_types)))
        if failures:return proof
        if epoch!=self._connection_epoch or not self.ib.isConnected():
            failures.append('EPOCH_CHANGED_BEFORE_SNAPSHOT');return proof
        try:
            # Capture the epoch only after acquiring the lock: an older waiter can
            # never use a newer epoch's responses.
            async with self.snapshot_lock:
                if epoch!=self._connection_epoch:
                    failures.append('EPOCH_CHANGED_AT_SNAPSHOT_LOCK');return proof
                positions=await asyncio.wait_for(self.ib.reqPositionsAsync(),request_timeout)
                if epoch!=self._connection_epoch:
                    failures.append('EPOCH_CHANGED_DURING_POSITIONS');return proof
                trades=await asyncio.wait_for(self.ib.reqAllOpenOrdersAsync(),request_timeout)
                if epoch!=self._connection_epoch:
                    failures.append('EPOCH_CHANGED_DURING_OPEN_ORDERS');return proof
                completed_method=getattr(self.ib,'reqCompletedOrdersAsync',None)
                if completed_method is None:
                    failures.append('COMPLETED_ORDERS_UNSUPPORTED');proof['retryable']=False;return proof
                completed=await asyncio.wait_for(completed_method(apiOnly=True),request_timeout)
                if epoch!=self._connection_epoch:
                    failures.append('EPOCH_CHANGED_DURING_COMPLETED_ORDERS');return proof
                own,other,ids=await asyncio.wait_for(self._execution_books(force=True),request_timeout)
                if epoch!=self._connection_epoch:
                    failures.append('EPOCH_CHANGED_DURING_EXECUTIONS');return proof
                values=await asyncio.wait_for(self.account_values(force=True),request_timeout)
                revision=self._book_revision
                await asyncio.sleep(stable_seconds)
                if epoch!=self._connection_epoch:
                    failures.append('EPOCH_CHANGED_DURING_STABILITY');return proof
                if revision!=self._book_revision:
                    failures.append('CALLBACKS_CHANGED_DURING_STABILITY');return proof
        except asyncio.TimeoutError as exc:
            failures.append(f'REQUEST_TIMEOUT:{type(exc).__name__}');return proof
        except Exception as exc:
            # Retry only explicit transport classes.  Semantic/validation
            # failures (for example non-positive account values) are
            # authoritative and must not be softened into reconnect retries.
            proof['retryable']=(not isinstance(exc,(PermissionError,ValueError))
                                and isinstance(exc,(ConnectionError,OSError)))
            failures.append(f'REQUEST_FAILED:{type(exc).__name__}:{exc}');return proof
        if not self.ib.isConnected():
            failures.append('DISCONNECTED_AFTER_SNAPSHOT');return proof
        # Broker-state reads can consume most of the pre-open budget.  Prices
        # that were fresh before them do not prove the connection is fresh now.
        now=datetime.now(timezone.utc)
        stale=sorted(k for k in needed
                     if k not in self.last_seen or self.last_seen[k]<started
                     or (now-self.last_seen[k]).total_seconds()>max_age)
        if stale:
            failures.append('STREAMS_STALE_AFTER_SNAPSHOT:'+','.join(stale));return proof
        bad_types=sorted(cid for cid in expected_contracts if self.data_types.get(cid)!=1)
        if bad_types:
            failures.append('MARKET_DATA_TYPE_CHANGED:'+','.join(map(str,bad_types)));return proof
        actual_client=getattr(getattr(self.ib,'client',None),'clientId',None)
        managed=list(self.ib.managedAccounts())
        if actual_client!=expected_client or self.config.account not in managed:
            failures.append('IDENTITY_CHANGED');proof['retryable']=False;return proof
        self._seed_recovery_signatures(positions,trades,values)
        token=(epoch,self._book_revision)
        # There is no await between the final epoch check and publishing health.
        self.healthy=True
        proof.update(complete=True,retryable=False,token=list(token),completed_orders=len(completed),
                     account_values=values,
                     stream_age_seconds={k:round((now-self.last_seen[k]).total_seconds(),3) for k in sorted(needed)},
                     snapshot=dict(positions=positions,trades=trades,own=dict(own),other=dict(other),
                                   own_exec_ids=sorted(ids)))
        self.diagnostic_callback(dict(kind='recovery_proof',epoch=epoch,token=list(token),
                                      restoration_code=self._restore_signal,complete=True,
                                      at=now.isoformat()))
        return proof

    async def history(self):
        import pandas as pd
        from ib_insync import util
        result={}
        for m in self.config.markets:
            rows=await self.ib.reqHistoricalDataAsync(self.contracts[m.signal.con_id],endDateTime='',durationStr='7 D',
                barSizeSetting='1 min',whatToShow='TRADES',useRTH=False,formatDate=2,keepUpToDate=False,timeout=90)
            frame=util.df(rows)
            if frame is None or frame.empty:raise ValueError('Historical futures minutes unavailable')
            frame['date']=pd.to_datetime(frame['date'],utc=True)
            result[m.name]=frame.set_index('date')
        return result

    async def range_history(self,duration='2 M',end=''):
        """Prior-range filter inputs: hourly TRADES bars for the signal contract and the expiry
        immediately before it (expired contracts via includeExpired). Read-only."""
        result={}
        for m in self.config.markets:
            # Per market: one market's failure is that market's UNAVAILABLE, never the other's.
            try:
                result[m.name]=await asyncio.wait_for(self._range_market(m,duration,end),200)
            except Exception as exc:
                result[m.name]=dict(error=f'{type(exc).__name__}: {exc}')
        return result

    async def _range_market(self,m,duration,end):
        import pandas as pd
        from ib_insync import Contract, util
        details=await self.ib.reqContractDetailsAsync(Contract(secType='FUT',symbol=m.signal.symbol,exchange='CME',
                                                                currency='USD',includeExpired=True))
        by={d.contract.lastTradeDateOrContractMonth[:8]:d.contract for d in details}
        if m.signal.expiry not in by or by[m.signal.expiry].conId!=m.signal.con_id:
            raise ValueError(f'{m.name}: signal contract not in the expiry list')
        earlier=[e for e in by if e<m.signal.expiry]
        if not earlier:raise ValueError(f'{m.name}: no earlier expiry for roll detection')
        contracts=[]
        for expiry in [max(earlier),m.signal.expiry]:
            c=by[expiry];c.includeExpired=True
            frame=None
            for attempt in range(2):
                # ib_insync returns an empty list on its own timeout: retry once, then report empty.
                rows=await self.ib.reqHistoricalDataAsync(c,endDateTime=end,durationStr=duration,barSizeSetting='1 hour',
                    whatToShow='TRADES',useRTH=False,formatDate=2,keepUpToDate=False,timeout=45)
                frame=util.df(rows)
                if frame is not None and not frame.empty:break
                if not attempt:await asyncio.sleep(2)
            if frame is None or frame.empty:
                if expiry==m.signal.expiry:raise ValueError(f'{m.name}: signal-contract range history unavailable')
                # Zero volume only if it expired before the window; inputs.prior_range enforces that.
                frame=pd.DataFrame(columns=['date','open','high','low','close','volume'])
            frame['date']=pd.to_datetime(frame['date'],utc=True)
            contracts.append(dict(expiry=expiry,con_id=c.conId,local_symbol=c.localSymbol,
                bars=frame.set_index('date')[['open','high','low','close','volume']].astype(float)))
        return dict(bar_minutes=60,contracts=contracts)

    def subscribe(self,on_tick,capture):
        self._on_tick=on_tick;self._capture=capture
        if not self._ticker_handler_installed:
            self.ib.pendingTickersEvent+=self._dispatch_ticks
            self._ticker_handler_installed=True
        self._request_subscriptions()

    def _request_subscriptions(self):
        if self._retired or not self.contracts or not self._ticker_handler_installed:return
        if self._subscription_epoch==self._connection_epoch and not self._needs_resubscribe:return
        self.ib.reqMarketDataType(1)
        for m in self.config.markets:
            self.ib.reqTickByTickData(self.contracts[m.signal.con_id],'Last',0,False)
            self.ib.reqTickByTickData(self.contracts[m.execution.con_id],'BidAsk',0,False)
            if self.config.entry_order_type=='stop_limit' and m.execution.con_id!=m.signal.con_id:
                # Normal Last updates suffice for the approximate shadow model;
                # avoid another tick-by-tick subscription/pacing limit per micro.
                self.ib.reqMktData(self.contracts[m.execution.con_id],'',False,False)
        self._subscription_epoch=self._connection_epoch;self._needs_resubscribe=False

    def _dispatch_ticks(self,tickers):self._ticks(tickers,self._on_tick,self._capture)

    def _ticks(self,tickers,on_tick,capture):
        for ticker in tickers:
            self.data_types[ticker.contract.conId]=ticker.marketDataType
            if ticker.marketDataType!=1:
                self._halt('NON_LIVE_MARKET_DATA');continue
            for t in ticker.tickByTicks:
                for m in self.config.markets:
                    if ticker.contract.conId==m.execution.con_id and hasattr(t,'bidPrice'):
                        self.quotes[m.name]=(t.bidPrice,t.askPrice,t.time)
                        self.last_seen[f'{m.name}:quote']=datetime.now(timezone.utc)
                        capture(dict(kind='quote',market=m.name,time=t.time.isoformat(),bid=t.bidPrice,ask=t.askPrice))
                    elif ticker.contract.conId==m.signal.con_id and hasattr(t,'price'):
                        self.last_seen[f'{m.name}:trade']=datetime.now(timezone.utc)
                        capture(dict(kind='trade',market=m.name,time=t.time.isoformat(),price=t.price))
                        if self.config.entry_order_type=='stop_limit' and m.execution.con_id==m.signal.con_id:
                            capture(dict(kind='execution_trade',market=m.name,time=t.time.isoformat(),price=t.price))
                        on_tick(m.name,t.time,t.price)
                    elif ticker.contract.conId==m.execution.con_id and hasattr(t,'price'):
                        capture(dict(kind='execution_trade',market=m.name,time=t.time.isoformat(),price=t.price))
            if self.config.entry_order_type=='stop_limit':
                for m in self.config.markets:
                    if ticker.contract.conId==m.execution.con_id and m.execution.con_id!=m.signal.con_id:
                        for t in ticker.ticks:
                            if t.tickType==4:
                                capture(dict(kind='execution_trade',market=m.name,time=t.time.isoformat(),price=t.price))

    def close(self):
        self._retired=True
        self.ib.disconnect()
