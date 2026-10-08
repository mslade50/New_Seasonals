"""Auditable local file installer. Never imports a trading worker or broker SDK.

Every command is a read-only plan unless --apply is explicit. CLI roots are fixed
by the pinned manifest. Only tests use an injected isolated root mapping.
"""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
import platform
import re
from pathlib import Path, PurePosixPath
import sqlite3
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

PACKAGE = Path(__file__).resolve().parent
SETTINGS = 'artifacts/open_breakout_runs/combined-deployment-settings.json'
RECEIPT = 'artifacts/open_breakout_runs/combined-install-receipt.json'
BACKUPS = 'artifacts/open_breakout_runs/combined-deployment-backups'
NY = ZoneInfo('America/New_York')
ID = 'open-breakout-reliability-20261007-v2'
LOCK = 'artifacts/open_breakout_runs/combined-deployment-install.lock'
FIXED_ROOTS = dict(repo=r'C:\Users\McKinley Slade\dev\New_Seasonals',
                   legend=r'C:\Users\McKinley Slade\OneDrive\trading_ibkr')

class Refusal(RuntimeError): pass

def digest(data): return hashlib.sha256(data).hexdigest()
def json_bytes(body): return (json.dumps(body, indent=2, sort_keys=True)+'\n').encode()
def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def file_hash(path): return digest(path.read_bytes()) if path.is_file() else None

def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def python_processes():
    # Names and IDs only. Never queries command lines, owners or executable paths.
    command = "@(Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | Select-Object Id,ProcessName) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                            capture_output=True, text=True, timeout=15)
    if result.returncode: raise Refusal('Cannot determine Python process IDs; refusing mutation')
    try: rows = json.loads(result.stdout) if result.stdout.strip() else []
    except ValueError: raise Refusal('Unknown Python process inventory')
    if isinstance(rows,dict): rows=[rows]
    return [r for r in rows if int(r['Id']) != os.getpid()]

class Deployment:
    def __init__(self, package=PACKAGE, *, fixture_roots=None, process_probe=None, now=None):
        self.package=Path(package).resolve()
        self.manifest=read_json(self.package/'deployment-manifest.json')
        if self.manifest.get('schema') != 1 or self.manifest.get('bundle_id') != ID:
            raise Refusal('Unknown deployment manifest')
        if self.manifest['roots'] != FIXED_ROOTS:
            raise Refusal('Manifest roots differ from reviewed DESKTOP-2KI41V6 paths')
        if self.manifest.get('default_disabled') is not True or self.manifest.get('native_close_qualification') is not None:
            raise Refusal('Unreviewed activation/close qualification in manifest')
        self.roots={k:Path(v).resolve() for k,v in (fixture_roots or self.manifest['roots']).items()}
        if set(self.roots) != {'repo','legend'} or self.roots['repo'] == self.roots['legend']:
            raise Refusal('Unknown/overlapping deployment roots')
        self.process_probe=process_probe or python_processes
        self.now=now or (lambda:datetime.now(NY))
        self.fixture=fixture_roots is not None
        self._validate_manifest()

    def target(self,root,name):
        if root not in self.roots: raise Refusal('Unknown target root')
        parts=PurePosixPath(name)
        if (parts.is_absolute() or not parts.parts or any(p in {'.','..'} or ':' in p or '\\' in p for p in parts.parts)):
            raise Refusal('Unsafe relative target path: '+name)
        path=self.roots[root].joinpath(*parts.parts)
        resolved=path.resolve()
        if not resolved.is_relative_to(self.roots[root]): raise Refusal('Target escapes root: '+name)
        for parent in [path]+list(path.parents):
            if parent == self.roots[root].parent: break
            if parent.exists() and (parent.is_symlink() or (parent.is_dir() and getattr(parent.stat(),'st_file_attributes',0)&0x400)):
                raise Refusal('Symlink/reparse deployment target: '+str(parent))
        return path

    def _validate_manifest(self):
        expected_repo={
            'open_breakout/config.py','open_breakout/ibkr.py','open_breakout/replay.py',
            'open_breakout/resting.py','open_breakout/service.py','open_breakout/price_health.py',
            'open_breakout/coordination.py','open_breakout/native_coordination.py','intraday_coordination.py',
            'legend/__init__.py','legend/coordination_bridge.py','open_breakout/__main__.py',
            'open_breakout/deployment_bootstrap.py',
            'open_breakout/order_reliability.py','open_breakout/order_policy.py','open_breakout/standby.py',
            'artifacts/open_breakout_runs/config-20260925-live.json',
            'artifacts/open_breakout_runs/config-20260925-shadow.json'}
        expected_legend={'legend_ema_fut.py','open_breakout_runtime_pointer.json'}
        rows=self.manifest['changes']
        actual={(r['root'],r['path']) for r in rows}
        if len(actual)!=len(rows) or actual != {('repo',n) for n in expected_repo}|{('legend',n) for n in expected_legend}:
            raise Refusal('Manifest target set differs from reviewed allowlist')
        for r in rows:
            self.target(r['root'],r['path'])
            payload=self.package/'payload'/r['root']/r['path']
            if file_hash(payload)!=r.get('payload_sha256',r['after_sha256']) or payload.stat().st_size!=r.get('payload_bytes',r['bytes']):
                raise Refusal('Payload hash/size changed: '+r['path'])
            if r.get('substitute_account') and (r['root'],r['path']) not in {
                ('legend','legend_ema_fut.py'),
                ('repo','artifacts/open_breakout_runs/config-20260925-live.json'),
                ('repo','artifacts/open_breakout_runs/config-20260925-shadow.json')}:
                raise Refusal('Unreviewed account template target')
        for r in self.manifest['prerequisites']: self.target(r['root'],r['path'])
        prerequisite_keys={(r['root'],r['path']) for r in self.manifest['prerequisites']}
        required_repo={'open_breakout/'+n+'.py' for n in
                       ('__init__','__main__','alerts','config','ibkr','inputs','replay','resting','service','standby','store','strategy')}
        required_repo|={'tests/test_open_breakout.py','config/open_breakout.example.json',
                        'broker_runtime/owner_connection.py','scripts/workspace_hygiene.py','daily_execution_report.py'}
        required_repo|={'artifacts/open_breakout_runs/'+n for n in
                       ('launch-live.ps1','launch-shadow.ps1','daily_launch.ps1','config-20260925-live.json','config-20260925-shadow.json')}
        required_legend={'execution_contracts.py','run_legend_ema_fut.bat','register_legend_ema_fut_task.ps1',
                         'legend_ema_fut.env','legend_ema_fut_enabled.flag','legend_ema_fut.py'}
        if (len(prerequisite_keys)!=len(self.manifest['prerequisites'])
                or prerequisite_keys!={('repo',n) for n in required_repo}|{('legend',n) for n in required_legend}):
            raise Refusal('Incomplete/duplicate production prerequisite inventory')
        for r in self.manifest['prerequisites']:
            if r['changed']!=((r['root'],r['path']) in actual): raise Refusal('Incorrect prerequisite change classification')

    def hygiene(self,command):
        if self.fixture: return  # Disposable test roots have no Git repository.
        args=[sys.executable,'-B',str(self.target('repo','scripts/workspace_hygiene.py')),command]
        if command=='start': args+=['--force']
        else:
            for r in self.manifest['changes']:
                if r['root']=='repo': args+=['--allow',r['path']]
        result=subprocess.run(args,cwd=self.roots['repo'],capture_output=True,text=True,timeout=60)
        if result.returncode: raise Refusal('Workspace hygiene failed: '+result.stdout+result.stderr)

    def payload_bytes(self,row):
        data=(self.package/'payload'/row['root']/row['path']).read_bytes()
        if row.get('substitute_account'):
            account=read_json(self.target('repo','artifacts/open_breakout_runs/config-20260925-live.json'))['account']
            if b'__LOCAL_BROKER_ACCOUNT__' not in data: raise Refusal('Missing account template token')
            data=data.replace(b'__LOCAL_BROKER_ACCOUNT__',account.encode('utf-8'))
        if digest(data)!=row['after_sha256'] or len(data)!=row['bytes']:
            raise Refusal('Rendered payload differs from reviewed production bytes: '+row['path'])
        return data

    @property
    def receipt_path(self): return self.target('repo',RECEIPT)
    @property
    def settings_path(self): return self.target('repo',SETTINGS)

    def base_checks(self):
        checks=[]
        for row in self.manifest['prerequisites']:
            if file_hash(self.target(row['root'],row['path'])) != row['sha256']:
                raise Refusal('Expected production prerequisite hash differs: '+row['path'])
            checks.append(row['path'])
        for row in self.manifest['changes']:
            target=self.target(row['root'],row['path'])
            if target.exists() and not target.is_file(): raise Refusal('Target is not a regular file: '+row['path'])
            if file_hash(target)!=row['before_sha256']: raise Refusal('Expected base hash/absence differs: '+row['path'])
            self.payload_bytes(row)
        if self.receipt_path.exists() or self.settings_path.exists():
            raise Refusal('Existing deployment receipt/settings; use installed preflight or rollback')
        live=read_json(self.target('repo','artifacts/open_breakout_runs/config-20260925-live.json'))
        legend={}
        for line in self.target('legend','legend_ema_fut.env').read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key,value=line.split('=',1); legend[key.strip()]=value.strip().strip('"').strip("'")
        if live['client_id']!=927481 or live['port']!=7496 or live['account']!=legend.get('LEGEND_EMA_FUT_ACCOUNT') or legend.get('LEGEND_EMA_FUT_CLIENT_ID')!='163':
            raise Refusal('Unexpected owner identity/account/port pairing')
        return checks

    def mutation_gate(self):
        if not self.fixture and platform.node().upper()!='DESKTOP-2KI41V6':
            raise Refusal('Mutation is qualified only for DESKTOP-2KI41V6')
        now=self.now().astimezone(NY)
        if time(7,30)<=now.time().replace(tzinfo=None)<time(16,15):
            raise Refusal('Writes allowed only 16:15-07:30 New York, outside scheduled strategy launch/trading window')
        processes=self.process_probe()
        if processes: raise Refusal('Other Python processes active; cannot prove workers quiescent from names/IDs: '+str(processes))
        return now

    @contextmanager
    def locked(self):
        lock=self.target('repo',LOCK)
        try: lock.mkdir()
        except FileExistsError: raise Refusal('Installation lock exists; do not remove without reviewing interrupted transaction')
        try: yield
        finally:
            # An unrecovered transaction remains locked, including a hard process exit.
            if not (lock/'transaction.json').exists():
                self.clear_lock(lock)

    def clear_lock(self,lock):
        if lock.resolve() != self.target('repo',LOCK).resolve(): raise Refusal('Unknown recovery lock')
        # Only this exact installer-owned directory, under its fixed runtime root.
        shutil.rmtree(lock)

    def installed(self):
        receipt=read_json(self.receipt_path)
        if receipt.get('bundle_id')!=ID or receipt.get('manifest_sha256')!=file_hash(self.package/'deployment-manifest.json'):
            raise Refusal('Installed receipt does not match this bundle')
        settings=read_json(self.settings_path)
        if settings.get('schema')!=1 or settings.get('bundle_id')!=ID or type(settings.get('enabled')) is not bool:
            raise Refusal('Unknown installed settings')
        if any(type(settings.get(k)) is not bool for k in ('price_enabled','order_reliability_enabled','coordination_enabled','native_automatic_close_enabled')) or settings['native_automatic_close_enabled']:
            raise Refusal('Unknown independent activation flags or automatic close')
        if digest(json_bytes(settings)) != receipt['settings_sha256']:
            raise Refusal('Settings changed outside installer')
        expected=receipt['installed_hashes']
        keys={r['root']+':'+r['path'] for r in self.manifest['changes']}
        if set(expected)!=keys or receipt.get('schema')!=1:
            raise Refusal('Incomplete installed receipt inventory')
        if (settings.get('repo_root')!=str(self.roots['repo'])
                or settings.get('coordination_db')!=str(self.target('repo','artifacts/open_breakout_runs/coordination/combined-ledger.sqlite'))
                or (not settings['enabled'] and settings.get('effective_from') is not None)):
            raise Refusal('Unknown installed runtime paths/session')
        for key,digest_value in expected.items():
            root,name=key.split(':',1)
            if file_hash(self.target(root,name))!=digest_value: raise Refusal('Installed file drift; refuses overwrite/rollback: '+name)
        for row in self.manifest['prerequisites']:
            if not row['changed'] and file_hash(self.target(row['root'],row['path']))!=row['sha256']:
                raise Refusal('Unchanged prerequisite drift: '+row['path'])
        return receipt,settings

    def report(self, installed=False):
        if installed:
            receipt,settings=self.installed()
            return dict(command='preflight',mode='installed',enabled=settings['enabled'],effective_from=settings['effective_from'],
                        hashes_verified=len(receipt['installed_hashes']),writes=0,broker_calls=0,worker_starts=0)
        checks=self.base_checks()
        return dict(command='preflight',mode='base',changes=self.manifest['changes'],prerequisites_verified=len(checks),
                    writes=0,broker_calls=0,worker_starts=0)

    def stage(self,apply=False):
        self.base_checks()
        if not apply: return dict(command='stage',apply=False,changes=self.manifest['changes'],enabled=False,worker_starts=0)
        self.mutation_gate()
        with self.locked():
            self.base_checks(); now=self.mutation_gate()
            self.hygiene('start')
            backups=self.target('repo',BACKUPS+'/'+ID)
            if backups.exists(): raise Refusal('Backup already exists; review interrupted/previous install')
            backups.mkdir(parents=True)
            original={}; writes={}; hashes={}
            for row in self.manifest['changes']:
                root,name=row['root'],row['path']; key=root+':'+name
                target=self.target(root,name)
                original[key]=target.read_bytes() if target.exists() else None
                if original[key] is not None:
                    backup=backups/root/name; atomic_write(backup,original[key])
                    if file_hash(backup)!=row['before_sha256']: raise Refusal('Backup verification failed')
                data=self.payload_bytes(row)
                if self.fixture and root=='legend' and name=='legend_ema_fut.py':
                    # Fixed-root startup guard, mapped only inside disposable test roots.
                    anchor=('_coord_expected_root = Path(r"'+FIXED_ROOTS['repo']+'")').encode()
                    replacement=('_coord_expected_root = Path('+repr(str(self.roots['repo']))+')').encode()
                    if data.count(anchor)!=1: raise Refusal('Fixture root guard anchor changed')
                    data=data.replace(anchor,replacement)
                if root=='legend' and name=='open_breakout_runtime_pointer.json':
                    data=json_bytes(dict(schema=1,repo_root=str(self.roots['repo'])))
                writes[target]=data; hashes[key]=digest(data)
            settings=dict(schema=1,bundle_id=ID,enabled=False,effective_from=None,repo_root=str(self.roots['repo']),
                          coordination_db=str(self.target('repo','artifacts/open_breakout_runs/coordination/combined-ledger.sqlite')),
                          runtime_hashes=[])
            settings.update(price_enabled=False,order_reliability_enabled=False,coordination_enabled=False,
                            native_automatic_close_enabled=False,reliability_qualification=None)
            writes[self.settings_path]=json_bytes(settings)
            receipt=dict(schema=1,bundle_id=ID,manifest_sha256=file_hash(self.package/'deployment-manifest.json'),
                         installed_utc=now.astimezone(timezone.utc).isoformat(),backup_relative=BACKUPS+'/'+ID,
                         installed_hashes=hashes,settings_sha256=digest(writes[self.settings_path]),
                         original_hashes={k:digest(v) if v is not None else None for k,v in original.items()},history=[])
            writes[self.receipt_path]=json_bytes(receipt)
            self.mutation_gate()
            self.transaction(writes, originals={self.target(*k.split(':',1)):v for k,v in original.items()},verify=lambda:self.hygiene('check'))
            return dict(command='stage',apply=True,enabled=False,receipt=str(self.receipt_path),backup=str(backups),
                        files=len(hashes),worker_starts=0,broker_calls=0)

    def transaction(self,writes,originals=None,verify=None):
        originals=dict(originals or {})
        for p in writes:
            if p not in originals: originals[p]=p.read_bytes() if p.exists() else None
        lock=self.target('repo',LOCK)
        if not lock.is_dir(): raise Refusal('Transaction requires installer lock')
        rows=[]
        for index,(p,data) in enumerate(writes.items()):
            root=next((k for k,v in self.roots.items() if p.is_relative_to(v)),None)
            if root is None: raise Refusal('Transaction target outside roots')
            before=originals[p]; backup=f'preimage-{index}.bin' if before is not None else None
            if backup: atomic_write(lock/backup,before)
            rows.append(dict(root=root,path=p.relative_to(self.roots[root]).as_posix(),backup=backup,
                             before_sha256=digest(before) if before is not None else None,
                             after_sha256=digest(data) if data is not None else None))
        atomic_write(lock/'transaction.json',json_bytes(dict(schema=1,bundle_id=ID,
                     manifest_sha256=file_hash(self.package/'deployment-manifest.json'),rows=rows)))
        touched=[]
        try:
            for p,data in writes.items():
                before=originals[p]
                if file_hash(p)!=(digest(before) if before is not None else None):
                    raise Refusal('Transaction target changed before write; preserve for manual review: '+str(p))
                touched.append(p)
                if data is None:
                    if p.exists(): p.unlink()
                else:
                    atomic_write(p,data)
                    if file_hash(p)!=digest(data): raise Refusal('Atomic write readback failed')
            if verify: verify()
        except BaseException:
            # No worker is started; restore complete pre-transaction bytes. Keep backups.
            for p in reversed(touched):
                data=originals[p]; intended=writes[p]
                if file_hash(p) not in {digest(data) if data is not None else None,digest(intended) if intended is not None else None}:
                    raise Refusal('Transaction target changed during restore; lock retained: '+str(p))
                if data is None:
                    if p.exists(): p.unlink()
                else: atomic_write(p,data)
                if file_hash(p)!=(digest(data) if data is not None else None):
                    raise Refusal('Transaction restore failed; lock retained; use recover')
            (lock/'transaction.json').unlink()
            raise
        (lock/'transaction.json').unlink()

    def recover(self,apply=False):
        """Restore interrupted file transaction only; never touches strategy state."""
        lock=self.target('repo',LOCK)
        if lock.is_dir() and not (lock/'transaction.json').exists():
            if any(not p.is_file() or not re.fullmatch(r'preimage-\d+\.bin',p.name) for p in lock.iterdir()):
                raise Refusal('Unjournalled lock has unknown files; preserve for manual review')
            if self.receipt_path.exists(): self.installed()
            else: self.base_checks()
            if not apply: return dict(command='recover',apply=False,files=0,cleanup_only=True,worker_starts=0,broker_calls=0)
            self.mutation_gate(); self.clear_lock(lock)
            return dict(command='recover',apply=True,files=0,cleanup_only=True,worker_starts=0,broker_calls=0)
        journal=read_json(lock/'transaction.json')
        if (journal.get('schema')!=1 or journal.get('bundle_id')!=ID
                or journal.get('manifest_sha256')!=file_hash(self.package/'deployment-manifest.json')):
            raise Refusal('Unknown recovery transaction')
        allowed={(r['root'],r['path']) for r in self.manifest['changes']}|{('repo',SETTINGS),('repo',RECEIPT),('repo',BACKUPS+'/'+ID+'/rollback-receipt.json')}
        writes={}
        for index,row in enumerate(journal['rows']):
            key=(row['root'],row['path'])
            if key not in allowed or self.target(*key) in writes: raise Refusal('Unknown/duplicate recovery target')
            target=self.target(*key)
            if file_hash(target) not in {row['before_sha256'],row['after_sha256']}:
                raise Refusal('Recovery target drift; manual preservation required: '+row['path'])
            data=None
            if row['before_sha256'] is not None:
                if row['backup']!=f'preimage-{index}.bin': raise Refusal('Unknown recovery backup')
                backup=self.target('repo',LOCK+'/'+row['backup'])
                if file_hash(backup)!=row['before_sha256']: raise Refusal('Recovery backup hash differs')
                data=backup.read_bytes()
            writes[target]=data
        if not apply: return dict(command='recover',apply=False,files=len(writes),worker_starts=0,broker_calls=0)
        self.mutation_gate()
        recovery=lock/'recovery.lock'
        try: recovery.mkdir()
        except FileExistsError: raise Refusal('Recovery already locked; inspect before retry')
        try:
            self.mutation_gate()
            # Revalidate before writes; a new unknown edit is never overwritten.
            self.recover(False)
            for target,data in reversed(list(writes.items())):
                if data is None:
                    if target.exists(): target.unlink()
                else: atomic_write(target,data)
                if file_hash(target)!=(digest(data) if data is not None else None):
                    raise Refusal('Recovery readback failed; journal retained')
            (lock/'transaction.json').unlink()
        finally: recovery.rmdir()
        self.clear_lock(lock)
        return dict(command='recover',apply=True,files=len(writes),worker_starts=0,broker_calls=0)

    def session_checks(self,session):
        value=date.fromisoformat(session); now=self.now().astimezone(NY)
        if not now.date()<value<=now.date()+timedelta(days=14): raise Refusal('Activation must be a future session within 14 days')
        import exchange_calendars as xcals
        import pandas as pd
        cal=xcals.get_calendar('XNYS'); label=pd.Timestamp(session)
        if not cal.is_session(label) or (cal.session_close(label)-cal.session_open(label)).total_seconds()!=23400:
            raise Refusal('Activation requires a full XNYS session')
        runs=self.target('repo','artifacts/open_breakout_runs')
        for p in runs.glob(session+'-*'):
            if p.is_dir() and (p/'runtime.sqlite').exists(): raise Refusal('Future session already has a journal; do not reset/rearm')
        ledger=self.target('repo','artifacts/open_breakout_runs/coordination/combined-ledger.sqlite')
        if ledger.exists():
            with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
                if db.execute('PRAGMA quick_check').fetchone()[0]!='ok': raise Refusal('Existing coordination ledger is corrupt')
                for scope,body in db.execute('SELECT scope, body FROM records'):
                    if json.loads(scope).get('day')==session:
                        raise Refusal('Future session already has coordination state; preserve/reconcile')

    def switch(self,enabled,session=None,confirmed=False,apply=False,*,reliability=False,qualification=None,paper_evidence=None,coordination=False):
        receipt,settings=self.installed()
        if reliability:
            if not enabled:raise Refusal('Reliability activation requires an enabled future-session rollout')
            try:
                import runpy
                validate_policy=runpy.run_path(str(self.package/'payload/repo/open_breakout/order_policy.py'))['validate_policy']
                qualification=validate_policy(qualification,'live')
            except Exception as exc:raise Refusal('Paper-qualified policy required: '+str(exc))
            if qualification['source_sha256']!=self.manifest['qualification_source_sha256']:
                raise Refusal('Qualification source hash differs from frozen payload inventory')
            if paper_evidence is None or not Path(paper_evidence).is_file() or file_hash(Path(paper_evidence))!=qualification['paper_evidence_sha256']:
                raise Refusal('Missing or changed native paper evidence file')
            proof=read_json(Path(paper_evidence))
            required_cases={'acceptance','cancel_race','late_partial_fill','protective_revision','restart_no_resend','disconnect_no_resend','owner_unknown_block','risk_atomicity'}
            if (proof.get('kind')!='paper-native-lifecycle' or proof.get('mode')!='paper'
                    or not str(proof.get('account','')).startswith('DU') or proof.get('operator_approved') is not True
                    or proof.get('source_sha256')!=self.manifest['qualification_source_sha256']
                    or any(proof.get('case_results',{}).get(k)!='PASS' for k in required_cases)
                    or proof.get('observed_ack_max_seconds')!=qualification['observed_ack_max_seconds']
                    or proof.get('observed_protection_ack_max_seconds')!=qualification['observed_protection_ack_max_seconds']):
                raise Refusal('Native paper evidence does not qualify the frozen source and required cases')
        elif qualification is not None:raise Refusal('Unexpected qualification for disabled reliability')
        if coordination and not enabled:raise Refusal('Coordination requires explicit future-session activation')
        if enabled:
            if settings['enabled']: raise Refusal('Already active; disable/review rather than overwrite activation')
            self.session_checks(session)
            if not confirmed: raise Refusal('Requires --confirm-fresh-owner-flat: explicit fresh exact-own-book manual reconciliation of both strategies')
        if not apply: return dict(command='activate' if enabled else 'deactivate',apply=False,enabled_after=enabled,session=session,worker_starts=0)
        self.mutation_gate()
        with self.locked():
            receipt,settings=self.installed(); now=self.mutation_gate()
            if enabled and settings['enabled']: raise Refusal('Already active after acquiring install lock')
            if enabled: self.session_checks(session)
            writes={}
            for name in ('config-20260925-live.json','config-20260925-shadow.json'):
                relative='artifacts/open_breakout_runs/'+name; p=self.target('repo',relative)
                cfg=read_json(p); cfg.update(allow_price_only_pause=enabled,price_outage_cancel_seconds=600,
                                            order_reliability_enabled=bool(reliability and enabled and name.endswith('-live.json')),
                                            order_reliability_policy=qualification if reliability and enabled and name.endswith('-live.json') else None)
                data=json_bytes(cfg); writes[p]=data; receipt['installed_hashes']['repo:'+relative]=digest(data)
            settings.update(enabled=enabled,effective_from=session if enabled else None)
            settings.update(price_enabled=enabled,order_reliability_enabled=bool(reliability and enabled),
                            coordination_enabled=bool(coordination and enabled),native_automatic_close_enabled=False,
                            reliability_qualification=qualification if reliability and enabled else None)
            settings['runtime_hashes']=[dict(path=str(self.target(*key.split(':',1))),sha256=value)
                                        for key,value in receipt['installed_hashes'].items()]
            # Pin original dependencies/launchers too, not only the changed files.
            settings['runtime_hashes'] += [dict(path=str(self.target(r['root'],r['path'])),sha256=r['sha256'])
                                          for r in self.manifest['prerequisites'] if not r['changed']]
            writes[self.settings_path]=json_bytes(settings)
            receipt['settings_sha256']=digest(writes[self.settings_path])
            receipt['history'].append(dict(action='activate' if enabled else 'deactivate',utc=now.astimezone(timezone.utc).isoformat(),
                                           session=session,operator_asserted_fresh_flat=confirmed if enabled else None))
            writes[self.receipt_path]=json_bytes(receipt)
            self.transaction(writes)
        return dict(command='activate' if enabled else 'deactivate',apply=True,enabled_after=enabled,
                    effective_from=settings['effective_from'],coordination_enabled=settings['coordination_enabled'],
                    order_reliability_enabled=settings['order_reliability_enabled'],
                    native_close_qualification=None,worker_starts=0,broker_calls=0)

    def rollback(self,apply=False):
        receipt,settings=self.installed(); writes={}
        # Only allow the installer-owned allowlist; verify every backup before changing anything.
        expected={r['root']+':'+r['path']:r['before_sha256'] for r in self.manifest['changes']}
        if receipt['original_hashes']!=expected: raise Refusal('Backup index differs from reviewed base hashes')
        if receipt['backup_relative']!=BACKUPS+'/'+ID: raise Refusal('Unknown backup directory')
        backups=self.target('repo',receipt['backup_relative'])
        for key,value in expected.items():
            root,name=key.split(':',1); target=self.target(root,name)
            data=None
            if value is not None:
                backup=backups/root/name
                if file_hash(backup)!=value: raise Refusal('Backup hash differs: '+name)
                data=backup.read_bytes()
            writes[target]=data
        if not apply: return dict(command='rollback',apply=False,files=len(writes),preserves='all journals/risk/attempts/ledger/flags',worker_starts=0)
        self.mutation_gate()
        with self.locked():
            self.installed(); self.mutation_gate()
            self.hygiene('start')
            # Restore source/config; remove only this installer's new settings/receipt.
            writes[self.settings_path]=None
            writes[self.receipt_path]=None
            writes[backups/'rollback-receipt.json']=json_bytes(dict(bundle_id=ID,utc=self.now().astimezone(timezone.utc).isoformat(),
                           restored=expected,preserved='journals/risk/attempts/ledger/flags/Scheduler'))
            self.transaction(writes,verify=lambda:self.hygiene('check'))
        return dict(command='rollback',apply=True,files=len(expected),worker_starts=0,broker_calls=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','stage','activate','deactivate','rollback','recover'])
    parser.add_argument('--apply',action='store_true',help='Explicit file writes only; never starts a worker')
    parser.add_argument('--installed',action='store_true',help='Preflight an installed disabled/active bundle')
    parser.add_argument('--session',help='Future full XNYS session for activation')
    parser.add_argument('--order-reliability',action='store_true',help='Requires a reviewed paper-qualified policy')
    parser.add_argument('--qualification',type=Path,help='Exact qualification JSON; offline fixtures are refused')
    parser.add_argument('--paper-evidence',type=Path,help='Frozen native paper evidence matching the policy hash')
    parser.add_argument('--coordination',action='store_true',help='Explicit separate Legend coordination rollout; automatic close remains disabled')
    parser.add_argument('--confirm-fresh-owner-flat',action='store_true',help='Operator confirms current exact-owned OpenBreakout+Legend executions/positions/orders are reconciled flat')
    args=parser.parse_args()
    try:
        verify_package(PACKAGE)
        deployment=Deployment()
        if args.command=='preflight': result=deployment.report(args.installed)
        elif args.command=='stage': result=deployment.stage(args.apply)
        elif args.command=='activate':
            if not args.session: raise Refusal('--session required')
            result=deployment.switch(True,args.session,args.confirm_fresh_owner_flat,args.apply,
                                     reliability=args.order_reliability,
                                     qualification=read_json(args.qualification) if args.qualification else None,
                                     paper_evidence=args.paper_evidence,
                                     coordination=args.coordination)
        elif args.command=='deactivate': result=deployment.switch(False,apply=args.apply)
        elif args.command=='recover': result=deployment.recover(args.apply)
        else: result=deployment.rollback(args.apply)
        print(json.dumps(result,indent=2))
    except (Refusal, OSError, ValueError, KeyError, sqlite3.Error) as error:
        print(json.dumps(dict(refused=True,reason=str(error),worker_starts=0,broker_calls=0)),flush=True)
        raise SystemExit(2)


def verify_package(package):
    package=Path(package).resolve()
    index=read_json(package/'package-integrity.json')
    if index.get('schema')!=1 or index.get('bundle_id')!=ID:
        raise Refusal('Unknown package integrity inventory')
    seen=set()
    for row in index['files']:
        name=row['path']; relative=PurePosixPath(name)
        if relative.is_absolute() or any(p in {'.','..'} or ':' in p or '\\' in p for p in relative.parts):
            raise Refusal('Unsafe package integrity path')
        path=package/name
        if name in seen or not path.resolve().is_relative_to(package): raise Refusal('Duplicate/escaping package inventory')
        seen.add(name)
        if file_hash(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise Refusal('Package integrity mismatch: '+name)
    expected={'installer.py','deployment-manifest.json','payload/repo/open_breakout/deployment_bootstrap.py'}
    if not expected<=seen: raise Refusal('Incomplete package integrity inventory')
    return len(seen)

if __name__=='__main__': main()
