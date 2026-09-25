"""IBKR transport. Importing this module never connects. Live routing requires the pilot config and session ack."""
import asyncio
from datetime import datetime, timezone
import math
from .config import LIVE_PILOT_MAX_CONTRACTS
from .service import TERMINAL, ACKNOWLEDGED

class IBKR:
    def __init__(self,config,session=None):
        from ib_insync import IB
        self.config=config;self.session=session;self.ib=IB();self.contracts={};self.quotes={};self.trades={}
        self.fill_callback=lambda *a:None
        self.status_callback=lambda *a:None
        self.halt_callback=lambda *a:None
        self.order_error_callback=lambda *a:None
        self.healthy=True;self.ack=None;self.contract_report={}
        self.last_seen={};self.data_types={}
        self.net_liq=self.excess=float('nan')
        # ib_insync keys openOrders/positions requests by a single name; overlapping calls orphan one.
        self.snapshot_lock=asyncio.Lock()

    async def connect(self):
        # Live: structural pilot limits plus the exact session/account acknowledgement.
        self.ack=self.config.authorize(self.session)
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
        self.ib.execDetailsEvent+=self._fill
        self.ib.orderStatusEvent+=self._status
        self.ib.disconnectedEvent+=lambda:self._halt('IB_DISCONNECTED')
        self.ib.errorEvent+=self._error

    def _halt(self,reason):
        self.healthy=False;self.halt_callback(reason)

    def _error(self,req_id,code,message,contract):
        if req_id in self.trades:
            self.order_error_callback(req_id,code,message)
        # 2103/2105 farm blips are warnings: stream staleness catches a real data outage.
        if code in {1100,1101,1102,1300,201,10147,10148,354,10167,10168,10089,10090,10189}:
            self._halt(f'IB_ERROR:{code}:{message}')

    def _fill(self,trade,fill):
        e=fill.execution
        if e.acctNumber==self.config.account and e.clientId==self.config.client_id:
            self.fill_callback(e.orderId,e.execId,e.shares,e.price)

    def _status(self,trade):
        if trade.order.account==self.config.account and trade.order.clientId==self.config.client_id:
            self.status_callback(trade.order.orderId,trade.orderStatus.status)

    def next_id(self):return self.ib.client.getReqId()

    def _order(self,oid,body):
        from ib_insync import Order
        fields=dict(orderId=oid,clientId=self.config.client_id,account=self.config.account,
                    action='BUY' if body['side']==1 else 'SELL',totalQuantity=body['qty'],
                    orderType=body['kind'],tif=body['tif'],orderRef=body['ref'],transmit=True,outsideRth=True)
        if 'limit' in body:fields['lmtPrice']=body['limit']
        if 'stop' in body:fields['auxPrice']=body['stop']
        if 'oca' in body:fields.update(ocaGroup=body['oca'],ocaType=2)
        if 'good_after' in body:fields['goodAfterTime']=body['good_after']
        return Order(**fields)

    def send(self,oid,market,body):
        self.config.authorize(self.session)
        if self.config.mode not in {'paper','live'}:
            raise PermissionError('Order transmission is available only in paper or authorized live mode')
        if body.get('account')!=self.config.account:
            raise PermissionError('Order account mismatch')
        if self.config.mode=='live' and not 0<body['qty']<=min(LIVE_PILOT_MAX_CONTRACTS,market.max_contracts):
            raise PermissionError('Live pilot quantity cap exceeded')
        # A halted feed still permits protective exits for actual executions.
        if body['kind']=='LMT' and (not self.healthy or not self.ib.isConnected()):
            raise RuntimeError('Unhealthy broker connection')
        order=self._order(oid,body)
        self.trades[oid]=self.ib.placeOrder(self.contracts[market.execution.con_id],order)

    def cancel(self,oid):
        self.config.authorize(self.session)
        if self.config.mode not in {'paper','live'}:
            raise PermissionError('Order cancellation is available only in paper or authorized live mode')
        trade=self.trades.get(oid)
        if trade is None:raise ValueError(f'Unknown order {oid}')
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

    async def account_values(self):
        rows=await self.ib.accountSummaryAsync()
        values={x.tag:float(x.value) for x in rows if x.account==self.config.account and x.currency=='USD' and x.tag in {'NetLiquidation','ExcessLiquidity'}}
        # connectAsync(account=...) already streams account updates; re-requesting them never
        # completes (verified 2026-09-24 on port 7496), so only fill gaps from that stream.
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

    async def snapshot(self):
        async with self.snapshot_lock:
            trades=await self.ib.reqAllOpenOrdersAsync()
            positions=await self.ib.reqPositionsAsync()
        return dict(positions={p.contract.conId:p.position for p in positions if p.account==self.config.account},
                    orders=[dict(id=t.order.orderId,client_id=t.order.clientId,con_id=t.contract.conId,ref=t.order.orderRef,
                        remaining=t.order.totalQuantity-t.orderStatus.filled,kind=t.order.orderType,
                        side=1 if t.order.action=='BUY' else -1,stop=t.order.auxPrice)
                            for t in trades if t.order.account==self.config.account])

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
        self.ib.reqMarketDataType(1)
        self.ib.pendingTickersEvent+=lambda tickers:self._ticks(tickers,on_tick,capture)
        for m in self.config.markets:
            self.ib.reqTickByTickData(self.contracts[m.signal.con_id],'Last',0,False)
            self.ib.reqTickByTickData(self.contracts[m.execution.con_id],'BidAsk',0,False)

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
                        on_tick(m.name,t.time,t.price)

    def close(self):self.ib.disconnect()
