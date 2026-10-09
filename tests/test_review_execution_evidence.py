"""Regress the actual raw normalization and patched entrypoint without sockets."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest

from broker_runtime import broker_reconciliation as B
from broker_runtime import prepare_review_execution as P
from broker_runtime import review_execution_runtime as R
from tests.test_agent_review_execution import broker_rows, prepared, command, TRADING_RUNTIME


class RawBroker:
    """Only read requests exist; mutation/connect methods deliberately absent."""
    def __init__(self, rows, *, missing=(), executions=()):
        self.live=[];self.past=[];self.executions=list(executions);self.reads=[]
        for row in rows:
            account,cid,client,oid,perm=row['identity']
            order=N(account=account,clientId=client,orderId=oid,permId=perm,
                    totalQuantity=row['qty'],action=row['action'],orderRef=row['ref'],
                    parentId=row['parent'],lmtPrice=row['limit'],auxPrice=row['stop'],
                    orderType=row['order_type'],ocaGroup=row['oca_group'],ocaType=row['oca_type'],
                    tif=row['tif'],goodAfterTime=row['good_after'],goodTillDate=row['good_till'],
                    outsideRth=row['outside_rth'],transmit=row['transmit'])
            terminal=row['status'] in B.TERMINAL
            if oid not in missing:order.filledQuantity=row['filled']
            # Real completedOrder supplies default zero in OrderStatus, even
            # when explicit cumulative quantity is absent on the Order.
            status=N(status=row['status'],filled=0 if terminal else row['filled'])
            trade=N(contract=N(conId=cid),order=order,orderStatus=status)
            (self.past if terminal else self.live).append(trade)
    def reqPositions(self):self.reads.append('positions');return []
    def reqAllOpenOrders(self):self.reads.append('open');return self.live
    def reqCompletedOrders(self,apiOnly=False):self.reads.append('completed');return self.past
    def reqExecutions(self):self.reads.append('executions');return self.executions


def raw_proof(tmp_path,monkeypatch,mutate,*,missing=(),execution_parent=False):
    g,_,_,_,plan,_=prepared(tmp_path)
    leg=plan['payload']['legs'][0];rows=broker_rows(leg,filled=10)
    mutate(rows)
    fills=[]
    if execution_parent:
        row=rows[0];account,cid,client,oid,perm=row['identity']
        fills=[N(contract=N(conId=cid),execution=N(acctNumber=account,clientId=client,
            orderId=oid,permId=perm,execId='qa.01',cumQty=10,shares=10,orderRef=row['ref']))]
    ib=RawBroker(rows,missing=[rows[i]['identity'][3] for i in missing],executions=fills)
    monkeypatch.setattr(R,'_capture',lambda g,ib,account,cid:B.capture(ib,account,cid))
    proof=R.evidence_leg(g,ib,plan['payload'],leg,{})
    assert ib.reads==['positions','open','completed','executions','positions','open']
    return proof,ib


def close_rows(rows):
    rows[1].update(status='Filled',filled=10)
    for row in rows[2:]:row.update(status='Cancelled',filled=0)


def test_raw_filled_status_never_mints_cumulative_quantity(tmp_path,monkeypatch):
    proof,ib=raw_proof(tmp_path,monkeypatch,lambda rows:None,missing=(0,))
    assert B.order_row(ib.past[0],completed=True)['filled'] is None
    assert proof['state']=='unknown' and 'entry fill' in proof['detail']


def test_explicit_execution_can_prove_missing_completed_entry_quantity(tmp_path,monkeypatch):
    proof,_=raw_proof(tmp_path,monkeypatch,lambda rows:None,missing=(0,),execution_parent=True)
    assert proof['state']=='filled' and proof['entry_filled']==10


@pytest.mark.parametrize('missing',[(1,),(2,),(3,)])
def test_missing_terminal_exit_quantity_never_mints_closed(tmp_path,monkeypatch,missing):
    proof,_=raw_proof(tmp_path,monkeypatch,close_rows,missing=missing)
    assert proof['state']=='unknown' and 'exit fill' in proof['detail']


@pytest.mark.parametrize('sibling',[2,3])
def test_matched_full_fills_with_working_exit_sibling_need_reconciliation(tmp_path,monkeypatch,sibling):
    def mutate(rows):close_rows(rows);rows[sibling]['status']='PreSubmitted'
    proof,_=raw_proof(tmp_path,monkeypatch,mutate)
    assert proof['state']=='unprotected' and 'siblings remain working' in proof['detail']


def test_matched_partial_fills_with_live_parent_remainder_need_reconciliation(tmp_path,monkeypatch):
    def mutate(rows):
        close_rows(rows);rows[0].update(status='Submitted',filled=3)
        rows[1].update(status='Cancelled',filled=3)
    proof,_=raw_proof(tmp_path,monkeypatch,mutate)
    assert proof['state']=='unprotected' and 'parent remains working' in proof['detail']


def test_partial_fills_can_close_only_after_entire_owned_chain_terminal(tmp_path,monkeypatch):
    def mutate(rows):
        close_rows(rows);rows[0].update(status='Cancelled',filled=3)
        rows[1].update(status='Cancelled',filled=3)
    proof,_=raw_proof(tmp_path,monkeypatch,mutate)
    assert proof['state']=='closed' and proof['entry_filled']==proof['exit_filled']==3


def test_full_fills_close_after_explicit_quantities_and_terminal_chain(tmp_path,monkeypatch):
    proof,_=raw_proof(tmp_path,monkeypatch,close_rows)
    assert proof['state']=='closed' and proof['entry_filled']==proof['exit_filled']==10


@pytest.mark.parametrize('source',[
    P.HERE.parent/'tests/fixtures/execution_runtime/execute_order_core.py',
    TRADING_RUNTIME/'execute_order.py'])
def test_patched_main_initializes_context_without_test_injecting_global(monkeypatch,source):
    if not source.exists():pytest.skip('external runtime absent')
    # Only parsed main is executed. Imports/top-level runtime code never run.
    tree=ast.parse(P.patch_executor(source.read_text(encoding='utf-8-sig')))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    cmd=command();g={'sys':N(argv=['offline-fixture',json.dumps(cmd)]),'json':json}
    observed=[]
    def adapter(env,payload):
        observed.append((env.get('_COMMAND_TYPE'),payload));return 'offline-adapter'
    monkeypatch.setattr(R,'run_executor',adapter)
    monkeypatch.setitem(sys.modules,'review_execution_runtime',R)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'patched-main-offline','exec'),g)
    assert '_COMMAND_TYPE' not in g
    assert g['main']()=='offline-adapter'
    assert observed==[('review_execution',cmd)]


def test_patched_normalizer_candidate_matches_corrected_source():
    # Source transform replaces only audited order_row, without importing broker.
    source=TRADING_RUNTIME/'broker_reconciliation.py'
    if not source.exists():pytest.skip('external dependency absent')
    node=next(n for n in ast.parse((P.HERE/'broker_reconciliation.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='order_row')
    text=(P.HERE/'broker_reconciliation.py').read_text();lines=text.splitlines(keepends=True)
    replacement=''.join(lines[node.lineno-1:node.end_lineno]).rstrip('\n')
    candidate=P.change_function(source.read_text(encoding='utf-8-sig'),'order_row',lambda _:replacement)
    assert ast.dump(ast.parse(candidate))==ast.dump(ast.parse(text))


@pytest.mark.parametrize('parent_status',['Cancelled','ApiCancelled'])
@pytest.mark.parametrize('child_status',['Submitted','PreSubmitted'])
def test_zero_fill_cancelled_parent_with_working_child_is_not_terminal(tmp_path,monkeypatch,parent_status,child_status):
    def mutate(rows):
        for row in rows:row.update(status='Cancelled',filled=0)
        rows[0]['status']=parent_status
        rows[2]['status']=child_status
    proof,_=raw_proof(tmp_path,monkeypatch,mutate)
    assert proof['state']=='unprotected' and 'children remain working' in proof['detail']
    from broker_runtime import review_execution as C
    assert C.summarize({'legs':[proof]})=='needs_reconciliation'


@pytest.mark.parametrize('status',['Cancelled','ApiCancelled'])
def test_zero_fill_cancellation_requires_entire_owned_chain_terminal(tmp_path,monkeypatch,status):
    def mutate(rows):
        for row in rows:row.update(status=status,filled=0)
    proof,_=raw_proof(tmp_path,monkeypatch,mutate)
    assert proof['state']=='cancelled' and proof['entry_filled']==proof['exit_filled']==0


def test_zero_fill_cancelled_chain_with_missing_child_quantity_stays_unknown(tmp_path,monkeypatch):
    def mutate(rows):
        for row in rows:row.update(status='Cancelled',filled=0)
    proof,_=raw_proof(tmp_path,monkeypatch,mutate,missing=(2,))
    assert proof['state']=='unknown' and 'exit fill quantity unavailable' in proof['detail']


def test_zero_fill_cancelled_parent_with_unknown_child_state_cannot_resolve(tmp_path,monkeypatch):
    def mutate(rows):
        for row in rows:row.update(status='Cancelled',filled=0)
        rows[2]['status']='Unknown'
    with pytest.raises(ValueError,match='order transition'):
        raw_proof(tmp_path,monkeypatch,mutate)
