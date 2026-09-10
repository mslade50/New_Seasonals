import copy
import datetime as dt
import pytest
import pandas as pd
from tests.test_tagged_inventory import seed, STRATEGY
from tagged_inventory import build_tagged_inventory
from closing_inventory import make_snapshot, read_snapshot, load_closing_inventory, snapshot_key
from actual_inventory_io import load_primary_nav, load_pending_entry_notionals
from olv_sizing import clip_quantity


def closing(session='2026-09-09'):
    at=pd.Timestamp(session+'T16:05:30',tz='America/New_York').tz_convert('UTC')
    opening=seed()
    opening['asof_utc']='2026-09-03T20:30:00Z'
    opening['positions'][0].update(symbol='SNA',entry_price=377.42,signed_qty=663)
    proof={'accounts':{'primary':dict(complete=True,broker_account='TEST_PRIMARY',
              continuous_from=opening['asof_utc'],complete_through=at.isoformat())}}
    book={'accounts':[dict(key='primary',broker_account='TEST_PRIMARY',error=None,nlv=605447.05,
                          orders_source_at=(at-pd.Timedelta(seconds=1)).timestamp(),orders=[],
                          positions=[dict(account='TEST_PRIMARY',con_id=42,position=663)])]}
    result=build_tagged_inventory(opening,[],proof,asof=at.isoformat(),algo_strategies={STRATEGY})
    assert result.status=='known'
    result.observed_book=book
    result.source_evidence=dict(seed=opening,fills=[],coverage=proof,book=book,entry_metadata={})
    return result,at,opening


@pytest.mark.parametrize('when',['2026-09-09T21:10:00Z','2026-09-10T08:15:00Z'])
def test_sna_uses_actual_broker_nav_and_prior_close_without_gateway(when):
    inventory,at,opening=closing()
    saved=make_snapshot(inventory,now=at)
    next_morning=read_snapshot(saved,now=when,reviewed_seed=opening)
    assert next_morning.source_kind=='prior_close'
    nav=load_primary_nav(next_morning,asof=next_morning.asof_utc)
    pending=load_pending_entry_notionals(next_morning,asof=next_morning.asof_utc)
    qty=clip_quantity(275,376.01,nav*.5,next_morning.notionals[('SNA',STRATEGY)]+pending.get(('SNA',STRATEGY),0))
    assert qty==139
    assert 663*377.42+qty*376.01<=nav*.5
    assert 663*377.42+(qty+1)*376.01>nav*.5
    assert clip_quantity(275,376.01,750000*.5,663*377.42)==275


@pytest.mark.parametrize('now', ['2026-09-11T08:15:00Z','2026-09-10T13:30:00Z','2026-09-09T20:01:00Z'])
def test_old_future_or_intraday_snapshot_is_rejected(now):
    inventory,at,opening=closing()
    with pytest.raises(ValueError):read_snapshot(make_snapshot(inventory,now=at),now=now,reviewed_seed=opening)


def test_prior_session_handles_weekend_and_labor_day():
    inventory,at,opening=closing('2026-09-04')
    # Use an earlier reviewed cutoff for this Friday fixture.
    opening['asof_utc']='2026-09-03T20:30:00Z'
    inventory.source_evidence['coverage']['accounts']['primary']['continuous_from']=opening['asof_utc']
    saved=make_snapshot(inventory,now=at)
    assert read_snapshot(saved,now='2026-09-08T08:15:00Z',reviewed_seed=opening).status=='known'


@pytest.mark.parametrize('mutate',[
    lambda s:s.update(observed_at='2026-09-09T19:59:00Z'),
    lambda s:s['evidence']['book']['accounts'][0].update(nlv=float('nan')),
    lambda s:s['evidence']['book']['accounts'][0].update(nlv=-1),
    lambda s:s['evidence']['book']['accounts'][0].update(broker_account='PA'),
    lambda s:s['evidence']['coverage']['accounts']['primary'].update(complete=False),
    lambda s:s['evidence']['book']['accounts'][0]['positions'][0].update(position=600),
    lambda s:s['evidence']['seed']['positions'][0].update(signed_qty=1),
])
def test_incomplete_or_incoherent_capture_rejected(mutate):
    inventory,at,opening=closing()
    saved=copy.deepcopy(make_snapshot(inventory,now=at));mutate(saved)
    with pytest.raises(ValueError):read_snapshot(saved,now='2026-09-10T08:15:00Z',reviewed_seed=opening)


def test_new_inventory_assignment_invalidates_older_capture():
    inventory,at,opening=closing();saved=copy.deepcopy(make_snapshot(inventory,now=at))
    opening['execution_allocations']={'manual':{'strategy':STRATEGY}}
    with pytest.raises(ValueError,match='attribution'):
        read_snapshot(saved,now='2026-09-10T08:15:00Z',reviewed_seed=opening)


@pytest.mark.parametrize('now',['2026-09-09T19:59:00Z','2026-09-09T20:04:59Z','2026-09-09T20:10:00Z'])
def test_capture_rejects_preclose_or_stale_observation(now):
    inventory,_,_=closing()
    with pytest.raises(ValueError):make_snapshot(inventory,now=now)


def test_missing_capture_is_explicitly_unknown_and_never_a_live_query(monkeypatch):
    import actual_inventory_io
    monkeypatch.setattr(actual_inventory_io,'load_actual_inventory',lambda **kw:pytest.fail('overnight broker query'))
    def missing(key):
        assert key==snapshot_key('2026-09-09')
        raise FileNotFoundError()
    result=load_closing_inventory(asof='2026-09-10T08:15:00Z',algo_strategies={STRATEGY},reader=missing)
    assert result.status=='unknown' and result.notionals=={}


def test_capture_is_local_only_at_1605_and_cannot_run_scan_or_orders():
    from scripts.automation_supervisor import CATALOG
    pipeline=CATALOG['inventory-close']
    assert pipeline.run_at_et==dt.time(16,5)
    job,=pipeline.jobs
    assert job.workflow is None and job.local_gate=='nyse_session'
    command,=job.commands
    assert command.argv==('{python}','scripts/capture_closing_inventory.py','--publish')


@pytest.mark.parametrize('nav,exempt,expected', [(605447.05,False,139),(750000,False,275),(605447.05,True,275)])
def test_actual_scanner_cap_block_uses_broker_nav_and_retains_etf_exemption(nav,exempt,expected):
    # Execute only the production sizing block, never run_daily_scan or I/O.
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'daily_scan.py').read_text(encoding='utf-8'))
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
               and ast.unparse(n.test).startswith('_tnc and shares > 0'))
    wrapper=ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),
        iter=ast.List(elts=[ast.Constant(0)],ctx=ast.Load()),body=[block],orelse=[])],type_ignores=[])
    inventory,_,_=closing()
    context=dict(_tnc={'pct_nav':.5,'exempt':['SNA'] if exempt else []},shares=275,
        _actual_inventory=inventory,_primary_nav=nav,_include_pending=True,_pending_capacity_known=True,
        t_clean='SNA',strat={'name':STRATEGY},open_notionals=inventory.notionals,
        _pending_notionals={},entry=377.42,_entry_offset_atr=.25,atr=5.629281,
        dist=5.629281*1.25,risk=0,sizing_note='',ACCOUNT_VALUE=750000,print=lambda *args:None)
    exec(compile(ast.fix_missing_locations(wrapper),'scanner-cap-only','exec'),context)
    assert context['shares']==expected


@pytest.mark.parametrize('partial,expected',[(False,'closing'),(True,'live')])
def test_scanner_uses_closing_capture_for_both_settled_bookends(partial,expected):
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'daily_scan.py').read_text(encoding='utf-8'))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_inventory_loader' for t in n.targets))
    ns=dict(is_intraday_partial=partial,load_actual_inventory='live',load_closing_inventory='closing')
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'inventory-source-only','exec'),ns)
    assert ns['_inventory_loader']==expected
