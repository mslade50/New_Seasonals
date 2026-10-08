"""Inert source-install/recovery and independent activation tests only."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import installer
from test_deployment import fixture,bootstrap,NY
from bridge_installer import BridgeDeployment,NAMES


def test_price_activation_keeps_Legend_and_reliability_disabled(fixture):
    d,roots,_=fixture;d.stage(True);d.switch(True,'2026-10-07',True,True)
    settings=d.installed()[1]
    assert settings['price_enabled'] and not settings['coordination_enabled']
    assert not settings['order_reliability_enabled'] and not settings['native_automatic_close_enabled']
    for name in ('live','shadow'):
        cfg=installer.read_json(roots['repo']/f'artifacts/open_breakout_runs/config-20260925-{name}.json')
        assert cfg['allow_price_only_pause'] and cfg['order_reliability_enabled'] is False
    from datetime import datetime
    m=bootstrap(d,roots);result=m.configure('legend',argv=[],now=datetime(2026,10,7,8,15,tzinfo=NY))
    import os
    assert result['enabled'] and not result['coordination_enabled']
    assert os.environ['INTRADAY_COORDINATION_ENABLED']=='0'


def test_no_offline_fixture_or_missing_native_paper_evidence_can_activate(fixture):
    d,_,_=fixture;d.stage(True)
    from payload.repo.open_breakout.order_policy import validate_policy
    p=dict(schema=1,scope='offline_fixture',qualification_id='inert',source_sha256=d.manifest['qualification_source_sha256'],
           soft_ack_seconds=3.,hard_ack_seconds=12.,cancel_seconds=15.,snapshot_seconds=5.,
           protection_ack_seconds=3.,observed_protection_ack_max_seconds=2.,observed_ack_max_seconds=8.,approved=True)
    with pytest.raises(installer.Refusal,match='Paper-qualified'):d.switch(True,'2026-10-07',True,True,reliability=True,qualification=p)
    p.update(scope='paper_qualified',paper_evidence_sha256='0'*64)
    with pytest.raises(installer.Refusal,match='evidence'):d.switch(True,'2026-10-07',True,True,reliability=True,qualification=p)
    assert not d.installed()[1]['enabled']


def test_paper_assertion_fixture_activates_live_only_and_keeps_shadow_legacy(fixture,tmp_path):
    # Synthetic installer parsing fixture, never a qualification claim.
    d,roots,_=fixture;d.stage(True)
    cases={k:'PASS' for k in ('acceptance','cancel_race','late_partial_fill','protective_revision',
                            'restart_no_resend','disconnect_no_resend','owner_unknown_block','risk_atomicity')}
    evidence=dict(kind='paper-native-lifecycle',mode='paper',account='DU_INERT_INSTALLER_FIXTURE',operator_approved=True,
                  source_sha256=d.manifest['qualification_source_sha256'],case_results=cases,observed_ack_max_seconds=8.,observed_protection_ack_max_seconds=2.)
    path=tmp_path/'inert-paper-assertion.json';path.write_text(json.dumps(evidence))
    p=dict(schema=1,scope='paper_qualified',qualification_id='INERT_INSTALLER_FIXTURE_ONLY',
           source_sha256=d.manifest['qualification_source_sha256'],soft_ack_seconds=3.,hard_ack_seconds=12.,
           cancel_seconds=15.,snapshot_seconds=5.,protection_ack_seconds=3.,observed_protection_ack_max_seconds=2.,observed_ack_max_seconds=8.,approved=True,
           paper_evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    d.switch(True,'2026-10-07',True,True,reliability=True,qualification=p,paper_evidence=path)
    live=installer.read_json(roots['repo']/'artifacts/open_breakout_runs/config-20260925-live.json')
    shadow=installer.read_json(roots['repo']/'artifacts/open_breakout_runs/config-20260925-shadow.json')
    assert live['order_reliability_enabled'] and shadow['order_reliability_enabled'] is False
    assert shadow['order_reliability_policy'] is None
    assert d.installed()[1]['coordination_enabled'] is False


@pytest.fixture
def bridge(tmp_path):
    root=tmp_path/'runtime';root.mkdir()
    source=Path(r'C:\Users\McKinley Slade\OneDrive\trading_ibkr/exec_agent.py')
    shutil.copyfile(source,root/'exec_agent.py')
    for name in ('seen.json','requests.json','execution-receipts.jsonl','schedules.json','agent.env'):
        (root/name).write_bytes(b'inert preserved sentinel')
    gates=[]
    d=BridgeDeployment(fixture_root=root,gate=lambda:gates.append('inert gate'))
    return d,root,gates


def test_bridge_drycheck_and_stage_rollback_preserve_all_request_receipts(bridge):
    d,root,gates=bridge;before=(root/'exec_agent.py').read_bytes()
    assert d.plan()['task_queries']==0 and not gates
    assert d.stage()['writes']==0 and not gates
    d.stage(True);assert len(gates)==2 and d.plan()['installed']
    for name in ('seen.json','requests.json','execution-receipts.jsonl','schedules.json','agent.env'):
        assert (root/name).read_bytes()==b'inert preserved sentinel'
    d.rollback(True);assert (root/'exec_agent.py').read_bytes()==before
    assert not (root/'execution_connection.py').exists()
    for name in ('seen.json','requests.json','execution-receipts.jsonl','schedules.json','agent.env'):
        assert (root/name).read_bytes()==b'inert preserved sentinel'


@pytest.mark.parametrize('crash_at',[1,2,3])
def test_bridge_interrupted_source_install_is_recoverable_without_resend(bridge,monkeypatch,crash_at):
    d,root,_=bridge;before=(root/'exec_agent.py').read_bytes();calls=[0];write=d._write
    def crash(path,data):
        calls[0]+=1
        if calls[0]==crash_at:raise OSError('inert source-install interruption')
        return write(path,data)
    monkeypatch.setattr(d,'_write',crash)
    with pytest.raises(OSError):d.stage(True)
    with pytest.raises(installer.Refusal,match='Interrupted'):d.plan()
    assert d.recover()['writes']==0
    d.recover(True)
    assert (root/'exec_agent.py').read_bytes()==before and not (root/'execution_connection.py').exists()
    assert not d.receipt.exists() and (root/'execution-receipts.jsonl').read_bytes()==b'inert preserved sentinel'


def test_bridge_unknown_target_and_drift_refuse(bridge):
    d,root,_=bridge
    with pytest.raises(installer.Refusal):d.target('seen.json')
    d.stage(True);(root/'execution_connection.py').write_text('unknown drift')
    with pytest.raises(installer.Refusal,match='drift'):d.rollback(True)


@pytest.mark.parametrize('other_python',[False,True])
def test_bridge_gate_excludes_installer_but_refuses_any_other_python(monkeypatch,other_python):
    import bridge_installer as module
    from datetime import datetime
    from types import SimpleNamespace
    class Clock:
        @staticmethod
        def now(zone):return datetime(2026,10,7,22,0,tzinfo=zone)
    monkeypatch.setattr(module,'datetime',Clock)
    commands=[]
    def query(args,**kwargs):
        script=args[-1];commands.append(script)
        if 'Get-Process' in script:
            assert 'Where-Object Id -ne '+str(module.os.getpid()) in script
            return SimpleNamespace(returncode=0,stdout='1' if other_python else '0')
        return SimpleNamespace(returncode=0,stdout='Ready')
    monkeypatch.setattr(module.subprocess,'run',query)
    if other_python:
        with pytest.raises(installer.Refusal,match='Python owner'):module.operator_gate()
        assert len(commands)==1
    else:
        module.operator_gate();assert len(commands)==2


def test_bridge_recovery_refuses_forged_transition_and_gate_time_drift(bridge,monkeypatch):
    d,root,_=bridge;original=d._write;calls=[0]
    def crash(path,data):
        calls[0]+=1
        if calls[0]==3:raise OSError('interrupted')
        original(path,data)
    monkeypatch.setattr(d,'_write',crash)
    with pytest.raises(OSError):d.stage(True)
    journal=d.lock/'transaction.json';saved=journal.read_bytes();body=json.loads(saved)
    body['rows'][0]['after_sha256']='0'*64;journal.write_text(json.dumps(body))
    with pytest.raises(installer.Refusal,match='transition'):d.recover(True)
    journal.write_bytes(saved)
    d.gate=lambda:(root/'exec_agent.py').write_bytes(b'concurrent unknown edit')
    with pytest.raises(installer.Refusal,match='after recovery gate'):d.recover(True)
    assert (root/'exec_agent.py').read_bytes()==b'concurrent unknown edit'
    assert d.lock.exists()


def test_bridge_state_reparse_point_refuses_before_any_write(bridge,monkeypatch):
    from types import SimpleNamespace
    d,root,_=bridge;d.state.mkdir()
    original=Path.stat
    def junction(path,*a,**kw):
        actual=original(path,*a,**kw)
        if path==d.state:return SimpleNamespace(st_mode=actual.st_mode,st_file_attributes=0x400)
        return actual
    monkeypatch.setattr(Path,'stat',junction)
    with pytest.raises(installer.Refusal,match='junction'):d.stage(True)
    assert not d.lock.exists() and not d.receipt.exists()
