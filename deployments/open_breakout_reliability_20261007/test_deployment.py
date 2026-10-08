"""Installer tests use disposable mapped roots, mock processes and mock launchers only."""
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo
import pytest
import installer

NY=ZoneInfo('America/New_York')
NOW=datetime(2026,10,6,18,0,tzinfo=NY)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def fixture(tmp_path):
    package=Path(__file__).resolve().parent
    manifest=json.loads((package/'deployment-manifest.json').read_text())
    roots={k:tmp_path/k for k in ('repo','legend')}
    for root in roots.values(): root.mkdir()
    # Small exact baseline fixture, not a checkout. No flags/credentials/journals copied.
    for row in manifest['prerequisites']:
        source=Path(manifest['roots'][row['root']])/row['path']
        target=roots[row['root']]/row['path']; target.parent.mkdir(parents=True,exist_ok=True)
        if row['path']=='legend_ema_fut.env':
            # Installer reads only the owner identity; private env values aren't copied.
            text='\n'.join(line for line in source.read_text(encoding='utf-8-sig').splitlines()
                           if line.startswith(('LEGEND_EMA_FUT_ACCOUNT=','LEGEND_EMA_FUT_CLIENT_ID=')))+'\n'
            target.write_text(text)
        elif row['path']=='legend_ema_fut_enabled.flag': target.write_text('fixture flag; inert')
        else: shutil.copyfile(source,target)
    for row in manifest['changes']:
        if row['before_sha256'] is not None:
            target=roots[row['root']]/row['path']
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(Path(manifest['roots'][row['root']])/row['path'],target)
    # Test package manifest pins sanitized env/flag byte fixtures, never production.
    test_package=tmp_path/'package'; shutil.copytree(package,test_package,
                         ignore=shutil.ignore_patterns('tests-report*','__pycache__','validation.json','.pytest_cache','evidence*','runtime_template'))
    for row in manifest['prerequisites']:
        if row['path'] in {'legend_ema_fut.env','legend_ema_fut_enabled.flag'}: row['sha256']=sha(roots[row['root']]/row['path'])
    (test_package/'deployment-manifest.json').write_text(json.dumps(manifest,indent=2))
    deployment=installer.Deployment(test_package,fixture_roots=roots,process_probe=lambda:[],now=lambda:NOW)
    return deployment,roots,test_package

def test_complete_disabled_stage_manifest_and_rollback(fixture):
    d,roots,p=fixture
    untouched=roots['repo']/'artifacts/open_breakout_runs/2026-10-06-live/runtime.sqlite'
    untouched.parent.mkdir(parents=True); untouched.write_bytes(b'journal remains opaque')
    risk=roots['repo']/'artifacts/open_breakout_runs/risk.json'; risk.write_bytes(b'risk attempts sentinel')
    d.report(); assert not d.receipt_path.exists()
    result=d.stage(True); assert result['files']==20 and not result['enabled']
    receipt,settings=d.installed(); assert settings['enabled'] is False
    for row in d.manifest['changes']:
        target=d.target(row['root'],row['path']); assert sha(target)==receipt['installed_hashes'][row['root']+':'+row['path']]
    assert 'Offline-only artifact' not in (roots['repo']/'open_breakout/__main__.py').read_text()
    assert 'Legend CLI is blocked' not in (roots['legend']/'legend_ema_fut.py').read_text()
    assert d.report(True)['enabled'] is False
    d.rollback(True)
    for row in d.manifest['changes']:
        assert installer.file_hash(d.target(row['root'],row['path']))==row['before_sha256']
    assert untouched.read_bytes()==b'journal remains opaque' and risk.read_bytes()==b'risk attempts sentinel'
    assert not d.receipt_path.exists() and not d.settings_path.exists()

def test_wrong_base_hash_refuses_without_writes(fixture):
    d,roots,p=fixture
    (roots['repo']/'open_breakout/service.py').write_text('unknown source')
    with pytest.raises(installer.Refusal,match='hash'): d.stage(True)
    assert not d.receipt_path.exists() and not (roots['repo']/installer.BACKUPS).exists()

@pytest.mark.parametrize('name',['../escape.py','/escape.py','C:/escape.py','open_breakout/../../escape.py','open_breakout\\escape.py'])
def test_wrong_path_refuses(fixture,name):
    d,roots,p=fixture
    with pytest.raises(installer.Refusal): d.target('repo',name)

def test_manifest_unreviewed_target_refuses(fixture):
    d,roots,p=fixture
    m=d.manifest; m['changes'][0]['path']='unrelated_strategy.py'
    (p/'deployment-manifest.json').write_text(json.dumps(m))
    with pytest.raises(installer.Refusal,match='allowlist'): installer.Deployment(p,fixture_roots=roots)

def test_unknown_new_file_refuses(fixture):
    d,roots,p=fixture
    (roots['repo']/'intraday_coordination.py').write_text('somebody else owns this')
    with pytest.raises(installer.Refusal,match='absence'): d.stage(True)

def test_active_process_refuses_without_commandline_inspection(fixture):
    d,roots,p=fixture; d.process_probe=lambda:[dict(Id=123,ProcessName='python')]
    with pytest.raises(installer.Refusal,match='Other Python'): d.stage(True)
    assert not d.receipt_path.exists()

def test_trading_window_refuses(fixture):
    d,roots,p=fixture; d.now=lambda:datetime(2026,10,6,9,0,tzinfo=NY)
    with pytest.raises(installer.Refusal,match='16:15'): d.stage(True)

def test_read_only_plans_ignore_running_processes(fixture):
    d,roots,p=fixture; d.process_probe=lambda:[dict(Id=123,ProcessName='python')]
    d.report(); assert d.stage()['apply'] is False; assert not d.receipt_path.exists()

def test_partial_write_failure_restores_all_preimages(fixture,monkeypatch):
    d,roots,p=fixture; original=(roots['repo']/'open_breakout/config.py').read_bytes()
    real=installer.atomic_write; count=[0]
    def fail_once(path,data):
        if not str(path).startswith(str(roots['repo']/installer.BACKUPS)):
            count[0]+=1
            if count[0]==3: raise OSError('injected write failure')
        return real(path,data)
    monkeypatch.setattr(installer,'atomic_write',fail_once)
    with pytest.raises(OSError,match='injected'): d.stage(True)
    assert (roots['repo']/'open_breakout/config.py').read_bytes()==original
    assert not d.settings_path.exists() and not (roots['repo']/'intraday_coordination.py').exists()

def test_activation_requires_fresh_manual_confirmation_and_future_session(fixture):
    d,roots,p=fixture; d.stage(True)
    with pytest.raises(installer.Refusal,match='confirm'): d.switch(True,'2026-10-07',False,True)
    with pytest.raises(installer.Refusal,match='future'): d.switch(True,'2026-10-06',True,True)
    with pytest.raises(installer.Refusal,match='full XNYS'): d.switch(True,'2026-10-10',True,True)
    assert d.installed()[1]['enabled'] is False

def test_activation_does_not_start_any_worker_and_rollback_preserves_ledger(fixture):
    d,roots,p=fixture; d.stage(True); result=d.switch(True,'2026-10-07',True,True,coordination=True)
    assert result['worker_starts']==0 and result['broker_calls']==0 and result['native_close_qualification'] is None
    for name in ('live','shadow'):
        assert json.loads((roots['repo']/f'artifacts/open_breakout_runs/config-20260925-{name}.json').read_text())['allow_price_only_pause'] is True
    ledger=roots['repo']/'artifacts/open_breakout_runs/coordination/combined-ledger.sqlite'
    ledger.parent.mkdir(parents=True); ledger.write_bytes(b'ledger sentinel; never erased')
    d.switch(False,apply=True); assert d.installed()[1]['enabled'] is False
    d.rollback(True); assert ledger.read_bytes()==b'ledger sentinel; never erased'

def test_future_journal_refuses_activation(fixture):
    d,roots,p=fixture; d.stage(True)
    journal=roots['repo']/'artifacts/open_breakout_runs/2026-10-07-live/runtime.sqlite'
    journal.parent.mkdir(); journal.write_bytes(b'future journal')
    with pytest.raises(installer.Refusal,match='journal'): d.switch(True,'2026-10-07',True,True,coordination=True)

def test_rollback_source_drift_refuses(fixture):
    d,roots,p=fixture; d.stage(True)
    target=roots['repo']/'open_breakout/service.py'; target.write_text('later owner edit')
    with pytest.raises(installer.Refusal,match='drift'): d.rollback(True)
    assert target.read_text()=='later owner edit'

def test_rollback_backup_tamper_refuses(fixture):
    d,roots,p=fixture; d.stage(True)
    backup=roots['repo']/installer.BACKUPS/installer.ID/'repo/open_breakout/service.py'; backup.write_text('tamper')
    with pytest.raises(installer.Refusal,match='Backup hash'): d.rollback(True)

def bootstrap(d,roots):
    spec=importlib.util.spec_from_file_location('fixture_bootstrap',roots['repo']/'open_breakout/deployment_bootstrap.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_disabled_bootstrap_overrides_inherited_activation(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); m=bootstrap(d,roots)
    monkeypatch.setenv('INTRADAY_COORDINATION_ENABLED','1'); monkeypatch.setenv('INTRADAY_COORDINATION_DB','C:/wrong.sqlite')
    assert m.configure('legend',argv=[])['enabled'] is False
    assert os.environ['INTRADAY_COORDINATION_ENABLED']=='0' and 'INTRADAY_COORDINATION_DB' not in os.environ

def test_active_live_workers_share_db_shadow_remains_disabled(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); d.switch(True,'2026-10-07',True,True,coordination=True); m=bootstrap(d,roots)
    now=datetime(2026,10,7,8,15,tzinfo=NY)
    a=m.configure('open_breakout',argv=['live-session','--session','2026-10-07'],now=now)
    b=m.configure('legend',argv=[],now=now); assert a['db']==b['db'] and a['enabled'] and b['enabled']
    assert m.configure('open_breakout',argv=['shadow-session'],now=now)['enabled'] is False
    assert not (roots['repo']/'artifacts/open_breakout_runs/coordination').exists()
    assert m.configure('legend',argv=['--kill'],now=NOW)['enabled'] is False
    assert m.configure('legend',argv=['--verify-only'],now=NOW)['enabled'] is False
    with pytest.raises(RuntimeError,match='future'): m.configure('legend',argv=[],now=NOW)

def test_active_bootstrap_refuses_changed_runtime_and_wrong_session(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); d.switch(True,'2026-10-07',True,True,coordination=True); m=bootstrap(d,roots)
    now=datetime(2026,10,7,8,15,tzinfo=NY)
    with pytest.raises(RuntimeError,match='current New York'): m.configure('open_breakout',argv=['live-session','--session','2026-10-06'],now=now)
    (roots['repo']/'open_breakout/service.py').write_text('changed')
    with pytest.raises(RuntimeError,match='source changed'): m.configure('legend',argv=[],now=now)

@pytest.mark.parametrize('active',[False,True])
def test_all_intended_imports_from_staged_copy_and_native_close_stays_unqualified(fixture,active):
    d,roots,p=fixture; d.stage(True)
    if active: d.switch(True,'2026-10-07',True,True,coordination=True)
    script=Path(__file__).resolve().parent/'validate_deployment.py'
    result=subprocess.run([sys.executable,'-B',str(script),'--probe-staged',str(roots['repo']),str(roots['legend'])]+(['--active'] if active else []),capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    body=json.loads(result.stdout); assert body['broker_network_attempts']==[] and len(body['imports'])>=19
    assert body['legend_file']==str(roots['legend']/'legend_ema_fut.py') and body['native_close_qualification'] is None

@pytest.mark.parametrize('active',[False,True])
def test_original_legend_batch_mocklaunch(fixture,active):
    d,roots,p=fixture; d.stage(True)
    if active: d.switch(True,'2026-10-07',True,True,coordination=True)
    source=roots['legend']/'legend_ema_fut.py'; full=source.read_text(); (roots['legend']/'legend_original_source.py').write_text(full)
    # Actual original BAT is exercised only against this inert substituted script.
    source.write_text('''import importlib.util, json, pathlib, sys
from unittest.mock import patch
from ib_insync import IB
def denied(*a,**k): raise AssertionError("broker prohibited")
with patch.object(IB,"connect",denied),patch.object(IB,"placeOrder",denied):
 p=pathlib.Path(__file__).parent/"legend_original_source.py"
 repo=pathlib.Path(json.loads(p.with_name("open_breakout_runtime_pointer.json").read_text())["repo_root"])
 sys.path.insert(0,str(repo))
 import open_breakout.deployment_bootstrap as bootstrap
 original=bootstrap.configure
 from datetime import datetime
 from zoneinfo import ZoneInfo
 bootstrap.configure=lambda worker:original(worker,now=datetime(2026,10,7,8,15,tzinfo=ZoneInfo("America/New_York")))
 s=importlib.util.spec_from_file_location("mock_legend_source",p)
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
 import os
 pathlib.Path(__file__).with_name("mock-launch-receipt.json").write_text(json.dumps(dict(enabled=os.environ["INTRADAY_COORDINATION_ENABLED"],file=str(m.SCRIPT_DIR),args=sys.argv[1:],started_trading=False)))
''')
    if active:
        # Fixture pins the inert substitution, never a changed production worker.
        receipt,settings=installer.read_json(d.receipt_path),installer.read_json(d.settings_path)
        key='legend:legend_ema_fut.py'; receipt['installed_hashes'][key]=sha(source)
        for row in settings['runtime_hashes']:
            if row['path']==str(source): row['sha256']=sha(source)
        receipt['settings_sha256']=installer.digest(installer.json_bytes(settings))
        d.settings_path.write_bytes(installer.json_bytes(settings)); d.receipt_path.write_bytes(installer.json_bytes(receipt))
    result=subprocess.run(['cmd.exe','/d','/c',str(roots['legend']/'run_legend_ema_fut.bat'),'--offline-mock'],capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads((roots['legend']/'mock-launch-receipt.json').read_text())
    assert receipt['enabled']==('1' if active else '0') and receipt['args']==['--offline-mock'] and not receipt['started_trading']

@pytest.mark.parametrize('active',[False,True])
@pytest.mark.parametrize('name,expected',[('launch-live.ps1','live-session'),('launch-shadow.ps1','shadow-session')])
def test_original_openbreakout_launcher_mock_child_environment(fixture,name,expected,active):
    d,roots,p=fixture; d.stage(True)
    if active: d.switch(True,'2026-10-07',True,True,coordination=True)
    # Only resolve the actual launcher's parsed argument/Start-Process contract; no original script execution.
    source=(roots['repo']/'artifacts/open_breakout_runs'/name).read_text()
    assert "@('-m', 'open_breakout', '"+expected+"'" in source
    assert '-WorkingDirectory $Repo -WindowStyle Hidden' in source
    assert '$Python' in source and str(Path(sys.executable)) in source
    script=p/'mock_ps_child.ps1'
    code="""param([string]$Python,[string]$Repo,[string]$Output,[string]$Mode)
$env:INTRADAY_COORDINATION_ENABLED='1'
$env:INTRADAY_COORDINATION_DB='C:\\wrong.sqlite'
$child=@('-B','-c', 'import json,os;from open_breakout.deployment_bootstrap import configure;configure("open_breakout",argv=["MODE"]);print(json.dumps(dict(enabled=os.environ["INTRADAY_COORDINATION_ENABLED"],db=os.environ.get("INTRADAY_COORDINATION_DB"),started_trading=False)))'.Replace('MODE',$Mode))
$proc=Start-Process -FilePath $Python -ArgumentList $child -WorkingDirectory $Repo -WindowStyle Hidden -RedirectStandardOutput $Output -RedirectStandardError ($Output+'.err') -Wait -PassThru
exit $proc.ExitCode
"""
    # Use a script file instead of -c quoting, retaining original Start-Process semantics.
    child=p/'mock_child.py'; child.write_text('''import json,os,sys
from open_breakout.deployment_bootstrap import configure
from datetime import datetime
from zoneinfo import ZoneInfo
configure("open_breakout",argv=[sys.argv[1],"--session","2026-10-07"],now=datetime(2026,10,7,8,15,tzinfo=ZoneInfo("America/New_York")))
print(json.dumps(dict(enabled=os.environ["INTRADAY_COORDINATION_ENABLED"],db=os.environ.get("INTRADAY_COORDINATION_DB"),started_trading=False)))
''')
    code=code[:code.index('$child=')]+"$child=@('-B', ('\"'+$Child+'\"'),$Mode)\n"+code[code.index('$proc='):]
    code=code.replace('[string]$Mode)', '[string]$Mode,[string]$Child)')
    script.write_text(code)
    output=p/(name+'.mock.json')
    env=dict(os.environ,PYTHONPATH=str(roots['repo']))
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-File',str(script),'-Python',sys.executable,'-Repo',str(roots['repo']),'-Output',str(output),'-Mode',expected,'-Child',str(child)],env=env,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr+(Path(str(output)+'.err').read_text() if Path(str(output)+'.err').exists() else '')
    receipt=json.loads(output.read_text()); enabled=active and expected=='live-session'
    assert receipt['enabled']==('1' if enabled else '0') and not receipt['started_trading']
    assert receipt['db']==(str(roots['repo']/'artifacts/open_breakout_runs/coordination/combined-ledger.sqlite') if enabled else None)


def test_manifest_root_and_prerequisite_tamper_refused(fixture):
    d,roots,p=fixture
    for kind in ('roots','prerequisites'):
        m=json.loads((p/'deployment-manifest.json').read_text())
        if kind=='roots': m['roots']['repo']=str(roots['repo'])
        else: m['prerequisites']=m['prerequisites'][:-1]
        (p/'deployment-manifest.json').write_text(json.dumps(m))
        with pytest.raises(installer.Refusal): installer.Deployment(p,fixture_roots=roots)
        (p/'deployment-manifest.json').write_text(json.dumps(d.manifest))


@pytest.mark.parametrize('active',[False,True])
def test_settings_tamper_blocks_live_but_safety_commands_remain_available(fixture,monkeypatch,active):
    d,roots,p=fixture; d.stage(True)
    if active: d.switch(True,'2026-10-07',True,True,coordination=True)
    m=bootstrap(d,roots); body=installer.read_json(d.settings_path)
    body['runtime_hashes']=[]; body['enabled']=True
    d.settings_path.write_bytes(installer.json_bytes(body))
    with pytest.raises(RuntimeError,match='changed deployment'): m.configure('legend',argv=[])
    assert os.environ['INTRADAY_COORDINATION_ENABLED']=='0'
    assert m.configure('legend',argv=['--kill'])['enabled'] is False
    with pytest.raises(installer.Refusal,match='Settings changed'): d.installed()


def test_empty_runtime_inventory_refused_even_if_receipt_hash_updated(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); d.switch(True,'2026-10-07',True,True,coordination=True)
    m=bootstrap(d,roots); body=installer.read_json(d.settings_path); receipt=installer.read_json(d.receipt_path)
    body['runtime_hashes']=[]; receipt['settings_sha256']=installer.digest(installer.json_bytes(body))
    d.settings_path.write_bytes(installer.json_bytes(body)); d.receipt_path.write_bytes(installer.json_bytes(receipt))
    with pytest.raises(RuntimeError,match='runtime inventory'):
        m.configure('legend',argv=[],now=datetime(2026,10,7,8,15,tzinfo=NY))


def test_missing_installed_receipt_inventory_refuses_rollback(fixture):
    d,roots,p=fixture; d.stage(True); receipt=installer.read_json(d.receipt_path)
    receipt['installed_hashes'].pop('repo:open_breakout/service.py')
    d.receipt_path.write_bytes(installer.json_bytes(receipt))
    with pytest.raises(installer.Refusal,match='inventory'): d.rollback(True)


def test_hygiene_failure_restores_sources(fixture,monkeypatch):
    d,roots,p=fixture; calls=[]
    def hygiene(command):
        calls.append(command)
        if command=='check': raise installer.Refusal('Workspace hygiene failed')
    d.hygiene=hygiene
    with pytest.raises(installer.Refusal,match='hygiene'): d.stage(True)
    assert calls==['start','check'] and not d.settings_path.exists()
    for row in d.manifest['changes']:
        assert installer.file_hash(d.target(row['root'],row['path']))==row['before_sha256']


@pytest.mark.parametrize('drift',[False,True])
def test_interrupted_transaction_recovery_preserves_unknown_edits(fixture,monkeypatch,drift):
    d,roots,p=fixture; before=(roots['repo']/'open_breakout/config.py').read_bytes()
    real=installer.atomic_write; failed=[False]
    def interrupted(path,data):
        if str(path)==str(roots['repo']/'open_breakout/ibkr.py'):
            failed[0]=True; raise OSError('simulated disk failure')
        if failed[0] and str(path)==str(roots['repo']/'open_breakout/config.py'):
            raise OSError('restore failure; recovery needed')
        return real(path,data)
    monkeypatch.setattr(installer,'atomic_write',interrupted)
    with pytest.raises(OSError): d.stage(True)
    monkeypatch.setattr(installer,'atomic_write',real)
    lock=roots['repo']/installer.LOCK; assert (lock/'transaction.json').exists()
    if drift:
        (roots['repo']/'open_breakout/config.py').write_bytes(b'later owner edit')
        with pytest.raises(installer.Refusal,match='drift'): d.recover(True)
        assert (roots['repo']/'open_breakout/config.py').read_bytes()==b'later owner edit'
    else:
        assert d.recover()['apply'] is False
        assert d.recover(True)['worker_starts']==0 and not lock.exists()
        assert (roots['repo']/'open_breakout/config.py').read_bytes()==before
        for row in d.manifest['changes']:
            assert installer.file_hash(d.target(row['root'],row['path']))==row['before_sha256']


def test_partial_activation_failure_restores_disabled_state(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); original_settings=d.settings_path.read_bytes()
    real=installer.atomic_write; failed=[False]
    def fail_once(path,data):
        if path==d.settings_path and not failed[0]:
            failed[0]=True; raise OSError('settings write failed')
        return real(path,data)
    monkeypatch.setattr(installer,'atomic_write',fail_once)
    with pytest.raises(OSError): d.switch(True,'2026-10-07',True,True,coordination=True)
    assert d.settings_path.read_bytes()==original_settings and d.installed()[1]['enabled'] is False


def test_partial_rollback_failure_restores_installed_state(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); real=installer.atomic_write; failed=[False]
    def fail_once(path,data):
        if path==roots['repo']/'open_breakout/ibkr.py' and not failed[0]:
            failed[0]=True; raise OSError('rollback write failed')
        return real(path,data)
    monkeypatch.setattr(installer,'atomic_write',fail_once)
    with pytest.raises(OSError): d.rollback(True)
    assert d.installed()[1]['enabled'] is False


def test_install_lock_blocks_live_startup_without_touching_halted_journal(fixture,monkeypatch):
    d,roots,p=fixture; d.stage(True); m=bootstrap(d,roots)
    journal=roots['repo']/'artifacts/open_breakout_runs/2026-10-06-live/runtime.sqlite'
    journal.parent.mkdir(); journal.write_bytes(b'latched halt attempts and orders')
    (roots['repo']/installer.LOCK).mkdir()
    with pytest.raises(RuntimeError,match='in progress'): m.configure('legend',argv=[])
    assert m.configure('legend',argv=['--verify-only'])['enabled'] is False
    assert journal.read_bytes()==b'latched halt attempts and orders'


def test_completed_transaction_leftover_lock_cleanup_only(fixture):
    d,roots,p=fixture; d.stage(True)
    lock=roots['repo']/installer.LOCK; lock.mkdir(); (lock/'preimage-0.bin').write_bytes(b'a retained transaction preimage')
    assert d.recover()['cleanup_only'] and d.recover(True)['files']==0
    assert not lock.exists() and d.installed()[1]['enabled'] is False


def test_package_integrity_refuses_changed_payload_and_incomplete_inventory(fixture):
    d,roots,p=fixture
    names=['installer.py','deployment-manifest.json','payload/repo/open_breakout/deployment_bootstrap.py']
    index=dict(schema=1,bundle_id=installer.ID,files=[dict(path=n,sha256=sha(p/n),bytes=(p/n).stat().st_size) for n in names])
    (p/'package-integrity.json').write_text(json.dumps(index))
    assert installer.verify_package(p)==3
    (p/names[-1]).write_text('tampered source')
    with pytest.raises(installer.Refusal,match='integrity mismatch'): installer.verify_package(p)
    index['files']=index['files'][:-1]; (p/'package-integrity.json').write_text(json.dumps(index))
    with pytest.raises(installer.Refusal,match='Incomplete'): installer.verify_package(p)


def test_source_edit_between_backup_and_write_is_preserved(fixture,monkeypatch):
    d,roots,p=fixture; real=installer.atomic_write; changed=[False]
    later=roots['repo']/'open_breakout/ibkr.py'
    def concurrent_edit(path,data):
        result=real(path,data)
        if path==roots['repo']/'open_breakout/config.py' and not changed[0]:
            changed[0]=True; later.write_bytes(b'concurrent owner edit')
        return result
    monkeypatch.setattr(installer,'atomic_write',concurrent_edit)
    with pytest.raises(installer.Refusal,match='changed before write'): d.stage(True)
    assert later.read_bytes()==b'concurrent owner edit' and not d.receipt_path.exists()


def test_external_legend_pointer_refuses_unreviewed_root_before_imports(fixture):
    d,roots,p=fixture; d.stage(True)
    pointer=roots['legend']/'open_breakout_runtime_pointer.json'
    pointer.write_text(json.dumps(dict(schema=1,repo_root=str(roots['repo'].parent/'wrong-runtime'))))
    script=Path(__file__).resolve().parent/'validate_deployment.py'
    result=subprocess.run([sys.executable,'-B',str(script),'--probe-staged',str(roots['repo']),str(roots['legend'])],capture_output=True,text=True,timeout=40)
    assert result.returncode!=0 and 'Exact reviewed OpenBreakout runtime root required before imports' in result.stderr
