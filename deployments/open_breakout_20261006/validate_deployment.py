"""Offline deployment validation. All staged worker imports run with native paths denied."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def probe(repo,legend,active=False):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(repo)); sys.path.insert(1,str(legend))
    import asyncio
    from ib_insync import IB
    from ib_insync.client import Client
    from ib_insync.connection import Connection
    import requests
    import smtplib
    import urllib.request
    import socket
    attempts=[]
    def denied(*a,**kw):
        attempts.append('native broker/network attempted')
        raise AssertionError('Offline import/mock launch forbids broker/network path')
    async def denied_async(*a,**kw): return denied(*a,**kw)
    IB.connect=denied; IB.connectAsync=denied_async; IB.placeOrder=denied; IB.cancelOrder=denied
    Client.connect=denied; Client.connectAsync=denied_async; Client.send=denied
    Connection.connectAsync=denied_async; asyncio.open_connection=denied_async; asyncio.start_server=denied_async
    requests.sessions.Session.request=denied; smtplib.SMTP.__init__=denied; urllib.request.urlopen=denied
    socket.create_connection=denied; socket.socket.connect=denied
    if active:
        from open_breakout import deployment_bootstrap as bootstrap
        from zoneinfo import ZoneInfo
        original=bootstrap.configure
        def active_configure(worker):
            return original(worker,argv=['live-session','--session','2026-10-07'] if worker=='open_breakout' else [],
                            now=datetime(2026,10,7,8,15,tzinfo=ZoneInfo('America/New_York')))
        bootstrap.configure=active_configure
    names=['open_breakout.'+p.stem for p in sorted((repo/'open_breakout').glob('*.py')) if p.stem!='__main__']
    names += ['open_breakout.__main__','intraday_coordination','legend.coordination_bridge','execution_contracts','broker_runtime.owner_connection']
    imports={}
    for name in names:
        module=importlib.import_module(name); imports[name]=str(Path(module.__file__).resolve())
        intended=repo if name.startswith(('open_breakout','intraday_coordination','legend.','broker_runtime')) else legend
        assert Path(module.__file__).resolve().is_relative_to(intended),name
    worker=legend/'legend_ema_fut.py'
    spec=importlib.util.spec_from_file_location('staged_legend_worker',worker)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    assert module.SCRIPT_DIR==legend
    assert module.ENV_PATH==legend/'legend_ema_fut.env' and module.JOURNAL_PATH==legend/'legend_ema_fut_journal.jsonl'
    assert module.ENABLE_FLAG==legend/'legend_ema_fut_enabled.flag'
    from open_breakout.native_coordination import NativeOwnerAdapter
    from types import SimpleNamespace
    adapter=NativeOwnerAdapter(SimpleNamespace(broker=SimpleNamespace(),coordination=None))
    assert adapter.qualification is None and not adapter.qualified()
    assert os.environ['INTRADAY_COORDINATION_ENABLED']==('1' if active else '0') and not attempts
    if active:
        assert os.environ['INTRADAY_COORDINATION_DB']==str(repo/'artifacts/open_breakout_runs/coordination/combined-ledger.sqlite')
    return dict(imports=imports,legend_file=str(worker),legend_runtime_paths_preserved=True,
                native_close_qualification=None,broker_network_attempts=attempts,worker_starts=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-staged',nargs=2,metavar=('REPO','LEGEND'))
    parser.add_argument('--active',action='store_true',help='Only used with disposable staged probe roots')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.probe_staged:
        print(json.dumps(probe(*[Path(p).resolve() for p in args.probe_staged],active=args.active),indent=2)); return
    import installer
    manifest=json.loads((ROOT/'deployment-manifest.json').read_text())
    output=(args.output or Path(manifest['roots']['repo'])/'artifacts/open_breakout_deployment').resolve()
    output.mkdir(parents=True,exist_ok=True)
    before={r['root']+':'+r['path']:sha(Path(manifest['roots'][r['root']])/r['path']) for r in manifest['prerequisites']}
    syntax=[]
    for path in ROOT.rglob('*.py'):
        if '__pycache__' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path)); syntax.append(path.relative_to(ROOT).as_posix())
    # Parse actual PS launchers without running them, including their Start-Process syntax.
    repo=Path(manifest['roots']['repo']); parsed={}
    for name in ('launch-live.ps1','launch-shadow.ps1','daily_launch.ps1'):
        path=repo/'artifacts/open_breakout_runs'/name
        command='$e=$null; $t=$null; [System.Management.Automation.Language.Parser]::ParseFile($env:OB_PARSE_PATH,[ref]$t,[ref]$e) | Out-Null; if ($e.Count) { $e | ConvertTo-Json; exit 2 }; "OK"'
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],env=dict(os.environ,OB_PARSE_PATH=str(path)),capture_output=True,text=True,timeout=20)
        if result.returncode or 'OK' not in result.stdout: raise RuntimeError(result.stdout+result.stderr)
        parsed[name]='PowerShell AST parsed; original launcher not executed'
    started=datetime.now(timezone.utc).isoformat()
    result=subprocess.run([sys.executable,'-B','-m','pytest','test_deployment.py','-q','--disable-warnings','--tb=short','-p','no:cacheprovider','--junitxml='+str(output/'tests-report.xml'),'--basetemp='+str(output/'pytest-fixtures')],cwd=ROOT,capture_output=True,text=True,timeout=300)
    (output/'tests-report.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    after={r['root']+':'+r['path']:sha(Path(manifest['roots'][r['root']])/r['path']) for r in manifest['prerequisites']}
    import xml.etree.ElementTree as ET
    cases=list(ET.parse(output/'tests-report.xml').iter('testcase'))
    counts=dict(tests=len(cases),passed=sum(not any(c.find(k) is not None for k in ('failure','error','skipped')) for c in cases),
                failed=sum(c.find('failure') is not None for c in cases),errors=sum(c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases))
    report=dict(started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),pytest_exit_code=result.returncode,
                counts=counts,syntax_files=syntax,launcher_static_analysis=parsed,
                production_hashes_unchanged=before==after,production_prerequisites=after,
                production_installer_applied=False,production_workers_started=False,production_broker_connections=0,
                scheduler_modified=False,native_close_qualification=None,
                mocked_launches_only=True,commands_do_not_start_workers=True)
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'production_prerequisites','syntax_files'}},indent=2))
    if result.returncode or before!=after: raise SystemExit(1)

if __name__=='__main__': main()
