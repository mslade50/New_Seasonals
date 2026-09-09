import ast
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from broker_runtime.prepare import patch_snapshot
from broker_runtime.prepare_olv_capacity import patch_book


@pytest.mark.parametrize('failure',[False,True])
def test_readonly_snapshot_attests_request_start_and_preserves_remaining(failure):
    path=Path(os.environ.get('IBKR_REVIEW_SOURCE','C:/Users/McKinley Slade/OneDrive/trading_ibkr'))/'book_snapshot.py'
    if not path.exists():pytest.skip('reviewed local broker source unavailable')
    source=path.read_text(encoding='utf-8-sig')
    if '"fills_source_session"' not in source:
        source=patch_snapshot(patch_book(source))
    tree=ast.parse(source)
    functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
    contract=NS(symbol='SPY',secType='STK',currency='USD',conId=42,lastTradeDateOrContractMonth='')
    order=NS(account='P',totalQuantity=100,action='SELL',orderType='LMT',lmtPrice=110,auxPrice=0,
             tif='GTC',goodAfterTime='',goodTillDate='',parentId=0,orderId=1,permId=2,
             orderRef='fixture',outsideRth=False,ocaGroup='g',ocaType=1,clientId=122)
    class IB:
        def connect(self,*a,**kw):assert kw['readonly'] is True
        def managedAccounts(self):return ['P']
        def reqAccountSummary(self):return []
        def accountSummary(self,*a):return [NS(account='P',tag='NetLiquidation',currency='USD',value='100000')]
        def portfolio(self,*a,**kw):return []
        def positions(self):return []
        def reqAllOpenOrders(self):return []
        def openTrades(self):return [NS(contract=contract,order=order,orderStatus=NS(status='Submitted',remaining=80,filled=20))]
        def reqExecutions(self):
            self.request_at=time.time()
            if failure:raise TimeoutError('fixture')
            return []
        def fills(self):return []
        def sleep(self,*a):pass
        def disconnect(self):pass
    broker=IB()
    ns={'IB':lambda:broker,'os':os,'time':time,'CONNECT_TIMEOUT':8}
    exec(compile(functions,'snapshot-fixture','exec'),ns)
    result=ns['snap_account'](dict(key='primary',label='Primary',host='localhost',port=0,cid=122))
    assert result['error'] is None
    assert result['orders'][0]['remaining']==80 and result['orders'][0]['filled']==20
    assert result['fills_complete'] is (not failure)
    if failure:assert result['fills_source_at'] is None
    else:assert result['orders_source_at']<=result['fills_source_at']/1000<=broker.request_at
