"""Deterministic quote-fill transport; a test/shadow model, not liquidity evidence."""
from datetime import time
from .service import own_ref
from .strategy import NY, aware

class SimBroker:
    def __init__(self,config,clock):
        self.config,self.clock=config,clock
        self.fill_callback=lambda *a:None;self.status_callback=lambda *a:None;self.halt_callback=lambda *a:None
        self.order_error_callback=lambda *a:None
        self.orders={};self.quotes={};self.positions={};self.serial=0;self.execution=0;self.error_codes={}
        # Own executions (conId, signed qty, orderRef); positions above is what OpenBreakout moved in the account.
        self.executions=[]
        # Another strategy in the same account: {conId: signed qty} with visible attributed executions,
        # and working order dicts in snapshot form (id, client_id, con_id, ref, remaining, kind, side, stop).
        self.foreign_positions={};self.foreign_orders=[]
        # executions_error: reqExecutions failed. foreign_visible=False: other clients' executions are not
        # returned to this client (the unconfirmed IB behaviour), so foreign positions show only in the account.
        self.executions_error=None;self.foreign_visible=True
    def last_error_code(self,oid):return self.error_codes.get(oid)
    def next_id(self):self.serial+=1;return self.serial
    async def equity(self):return self.config.shadow_equity
    async def check_margin(self,*args):return
    def quote(self,name):return self.quotes[name]
    def status(self,oid):return self.orders.get(oid,{}).get('status','UNKNOWN')
    def cancel(self,oid):
        order=self.orders.get(oid)
        if order and order['status']=='Submitted':order['status']='Cancelled';self.status_callback(oid,'Cancelled')
    async def wait_terminal(self,oid,timeout):
        if self.status(oid) not in {'Filled','Cancelled'}:raise TimeoutError('Simulated entry is not terminal')
    async def wait_ack(self,oid,timeout):
        if self.status(oid) not in {'Submitted','Filled'}:raise TimeoutError('Simulated protection not acknowledged')
    def send(self,oid,market,body):
        self.orders[oid]=dict(body=body.copy(),market=market,status='Submitted')
        self.status_callback(oid,'Submitted')
        if body['kind']=='LMT':
            bid,ask,_=self.quote(market.name)
            fill=ask if body['side']==1 else bid
            if (fill-body['limit'])*body['side']<=0:self.execute(oid,body['qty'],fill)
            else:self.orders[oid]['status']='Cancelled';self.status_callback(oid,'Cancelled')
        elif body['kind']=='MKT' and 'good_after' not in body:
            bid,ask,_=self.quote(market.name)
            self.execute(oid,body['qty'],ask if body['side']==1 else bid)
    def execute(self,oid,qty,price):
        order=self.orders[oid];body=order['body'];cid=order['market'].execution.con_id
        self.execution+=1
        self.record(cid,body['side']*qty,body['ref'],f'SIM-{self.execution}')
        self.fill_callback(oid,f'SIM-{self.execution}',qty,price)
        order['status']='Filled';self.status_callback(oid,'Filled')
        if 'oca' in body:
            for other_id,other in list(self.orders.items()):
                if other_id!=oid and other['body'].get('oca')==body['oca'] and other['status']=='Submitted':
                    other['status']='Cancelled';self.status_callback(other_id,'Cancelled')
    def update_quote(self,name,bid,ask,stamp):
        self.quotes[name]=(bid,ask,aware(stamp))
        for oid,order in list(self.orders.items()):
            if order['market'].name!=name or order['status']!='Submitted':continue
            body=order['body'];price=bid if body['side']==-1 else ask
            triggered=body['kind']=='STP' and (price-body['stop'])*body['side']>=0
            timed=body['kind']=='MKT' and aware(stamp).astimezone(NY).time()>=time(15,55)
            if triggered or timed:self.execute(oid,body['qty'],price)
    def record(self,cid,signed,ref,exec_id=None):
        self.positions[cid]=self.positions.get(cid,0)+signed
        self.executions.append((cid,signed,ref,exec_id))
    def _own(self):
        own={}
        for cid,signed,ref,_ in self.executions:
            if own_ref(ref):own[cid]=own.get(cid,0)+signed
        return own
    def _own_ids(self):
        return sorted(x for _,_,ref,x in self.executions if x and own_ref(ref))
    async def own_position(self,con_id):
        if self.executions_error:raise TimeoutError(self.executions_error)
        return self._own().get(con_id,0)
    async def snapshot(self):
        account={cid:self.positions.get(cid,0)+self.foreign_positions.get(cid,0) for cid in {*self.positions,*self.foreign_positions}}
        # executions_error simulates a failed/timed-out reqExecutions: own position UNKNOWN (None).
        failed=bool(self.executions_error)
        return dict(positions=account,own=None if failed else self._own(),foreign=None if failed else (self.foreign_positions.copy() if self.foreign_visible else {}),
            own_exec_ids=None if failed else self._own_ids(),executions_error=self.executions_error,
            orders=[dict(id=oid,client_id=self.config.client_id,
            con_id=o['market'].execution.con_id,ref=o['body']['ref'],remaining=o['body']['qty'],
            kind=o['body']['kind'],side=o['body']['side'],stop=o['body'].get('stop',0)) for oid,o in self.orders.items() if o['status']=='Submitted']
            +[dict(o) for o in self.foreign_orders])
