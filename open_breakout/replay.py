"""Deterministic quote-fill transport; a test/shadow model, not liquidity evidence."""
from datetime import time
from .strategy import NY, aware

class SimBroker:
    def __init__(self,config,clock):
        self.config,self.clock=config,clock
        self.fill_callback=lambda *a:None;self.status_callback=lambda *a:None;self.halt_callback=lambda *a:None
        self.order_error_callback=lambda *a:None
        self.orders={};self.quotes={};self.positions={};self.serial=0;self.execution=0;self.error_codes={}
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
        self.positions[cid]=self.positions.get(cid,0)+body['side']*qty
        self.execution+=1
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
    async def snapshot(self):
        return dict(positions=self.positions.copy(),orders=[dict(id=oid,client_id=self.config.client_id,
            con_id=o['market'].execution.con_id,ref=o['body']['ref'],remaining=o['body']['qty'],
            kind=o['body']['kind'],side=o['body']['side'],stop=o['body'].get('stop',0)) for oid,o in self.orders.items() if o['status']=='Submitted'])
