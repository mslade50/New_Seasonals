"""Separate two-file reconnect installer. Default is read-only; no worker/broker actions.

Only the future operator --apply path queries ExecAgent readiness. Tests inject
an inert gate and mapped roots. Receipts, requests, schedules, credentials and
strategy state are outside the fixed source-file allowlist.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from installer import atomic_write, json_bytes, file_hash, Refusal

ROOT=Path(__file__).resolve().parent
RUNTIME=Path(r'C:\Users\McKinley Slade\OneDrive\trading_ibkr')
NAMES=('execution_connection.py','exec_agent.py')
ID='execution-bridge-reconnect-20261007-v2'


def operator_gate():
    now=datetime.now(ZoneInfo('America/New_York'))
    minutes=now.hour*60+now.minute
    if not (minutes>=21*60+5 or minutes<4*60+55):
        raise Refusal('Bridge installation requires normal exited-agent window21:05–04:55 New York')
    # Conservative names/IDs-only inventory. A detached/manual Python owner
    # cannot be distinguished without command lines, so any Python blocks apply.
    inventory=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "@(Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | Where-Object Id -ne "+str(os.getpid())+").Count"],
        text=True,capture_output=True,timeout=15)
    if inventory.returncode or inventory.stdout.strip()!='0':
        raise Refusal('Python owner may still run; wait for normal quiescence, do not stop/restart it')
    # No command lines, settings mutation, starts or restarts.
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "(Get-ScheduledTask -TaskName ExecAgent -TaskPath '\\' -ErrorAction Stop).State"],
        text=True,capture_output=True,timeout=15)
    if result.returncode or result.stdout.strip()!='Ready':
        raise Refusal('ExecAgent must be Ready after its normal exit; do not stop/restart it')


class BridgeDeployment:
    def __init__(self,candidate=None,*,fixture_root=None,gate=None):
        self.candidate=Path(candidate or ROOT/'execution-bridge/candidate').resolve()
        self.root=Path(fixture_root or RUNTIME).resolve()
        self.gate=gate or operator_gate
        self.manifest=json.loads((self.candidate/'manifest.json').read_text())
        if set(self.manifest['files'])!=set(NAMES) or self.manifest.get('installed') is not False:
            raise Refusal('Unknown bridge candidate manifest')
        if self.manifest['source_agent_sha256']!='7ec1f52ca49ea66de4504996e4889536fef5494e4ea490ca5312f8b5490bd817':
            raise Refusal('Unreviewed bridge base')
        for name in NAMES:
            if file_hash(self.candidate/name)!=self.manifest['files'][name]:raise Refusal('Bridge candidate hash differs')
        self.state=self.checked(self.root/'.reliability-reconnect-source')
        self.lock=self.checked(self.state/'install.lock')
        self.receipt=self.checked(self.state/'receipt.json')

    def checked(self,path):
        path=Path(path)
        if not path.is_relative_to(self.root) or not path.resolve().is_relative_to(self.root):
            raise Refusal('Bridge path escapes reviewed runtime')
        for ancestor in (path,*path.parents):
            if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.stat(),'st_file_attributes',0)&0x400):
                raise Refusal('Bridge path traverses symlink/junction')
            if ancestor==self.root:break
        return path

    def target(self,name):
        if name not in NAMES:raise Refusal('Bridge target outside fixed allowlist')
        path=self.checked(self.root/name)
        if path.is_symlink() or path.resolve().parent!=self.root:raise Refusal('Bridge target escapes reviewed root')
        return path

    def plan(self):
        for path in (self.state,self.lock,self.receipt,self.state/'agent-before.py'):self.checked(path)
        if self.lock.exists():raise Refusal('Interrupted bridge source transaction; run recover plan')
        if self.receipt.exists():
            receipt=json.loads(self.receipt.read_text())
            if receipt.get('id')!=ID or self.receipt.read_bytes()!=self.receipt_bytes():raise Refusal('Unknown bridge receipt')
            for name in NAMES:
                if file_hash(self.target(name))!=receipt['after'][name]:raise Refusal('Installed bridge source drift')
            return dict(command='preflight',installed=True,writes=0,worker_starts=0,broker_calls=0,
                        task_queries=0,receipt=str(self.receipt))
        if file_hash(self.target('exec_agent.py'))!=self.manifest['source_agent_sha256']:
            raise Refusal('Bridge runtime agent differs from reviewed baseline')
        if self.target('execution_connection.py').exists():raise Refusal('Unknown preexisting reconnect helper')
        return dict(command='stage',installed=False,writes=0,planned_files=list(NAMES),
                    worker_starts=0,broker_calls=0,task_queries=0,receipts_preserved=True)

    def _write(self,path,data):atomic_write(self.checked(path),data)

    def receipt_bytes(self):
        return json_bytes(dict(id=ID,before={'exec_agent.py':self.manifest['source_agent_sha256'],'execution_connection.py':None},
                     after=self.manifest['files'],worker_starts=0,broker_calls=0,
                     requests_receipts_schedules_credentials_preserved=True))

    def transitions(self,operation):
        receipt_sha=hashlib.sha256(self.receipt_bytes()).hexdigest()
        original=self.manifest['source_agent_sha256']
        if operation=='stage':
            return {'agent-before.py':(None,original),'execution_connection.py':(None,self.manifest['files']['execution_connection.py']),
                    'exec_agent.py':(original,self.manifest['files']['exec_agent.py']),'receipt.json':(None,receipt_sha)}
        if operation=='rollback':
            return {'exec_agent.py':(self.manifest['files']['exec_agent.py'],original),
                    'execution_connection.py':(self.manifest['files']['execution_connection.py'],None),'receipt.json':(receipt_sha,None)}
        raise Refusal('Unknown bridge transaction operation')

    def row_target(self,name):
        if name in NAMES:return self.target(name)
        if name in {'receipt.json','agent-before.py'}:return self.checked(self.state/name)
        raise Refusal('Unknown bridge transaction path')

    def _clear_lock(self):
        self.checked(self.lock)
        if self.lock.resolve()!=self.root/'.reliability-reconnect-source/install.lock' or self.lock.is_symlink():
            raise Refusal('Bridge lock escapes reviewed runtime')
        for p in self.lock.iterdir():
            if not p.is_file() or p.is_symlink() or not (p.name=='transaction.json' or p.name.startswith('preimage-') and p.name.endswith('.bin')):
                raise Refusal('Unknown bridge lock contents; preserve for review')
        for p in self.lock.iterdir():p.unlink()
        self.lock.rmdir()

    def _transaction(self,writes,operation):
        self.checked(self.state);self.checked(self.lock)
        self.state.mkdir(parents=True,exist_ok=True)
        try:self.lock.mkdir()
        except FileExistsError:raise Refusal('Bridge installer locked')
        self.gate()
        rows=[]
        transitions=self.transitions(operation)
        if {p.name for p,_ in writes}!=set(transitions):raise Refusal('Incomplete bridge source transaction')
        # Baselines and durable rollback backup are checked under the lock.
        for path,data in writes:
            self.checked(path)
            before_sha,after_sha=transitions[path.name]
            if file_hash(path)!=before_sha or (hashlib.sha256(data).hexdigest() if data is not None else None)!=after_sha:
                raise Refusal('Bridge baseline changed under transaction lock')
        for i,(path,data) in enumerate(writes):
            before=path.read_bytes() if path.exists() else None
            name=f'preimage-{i}.bin'
            if before is not None:atomic_write(self.lock/name,before)
            rows.append(dict(path=path.name,before_sha256=hashlib.sha256(before).hexdigest() if before is not None else None,
                             after_sha256=hashlib.sha256(data).hexdigest() if data is not None else None,backup=name))
        atomic_write(self.checked(self.lock/'transaction.json'),json_bytes(dict(id=ID,operation=operation,rows=rows)))
        try:
            for path,data in writes:
                self.checked(path)
                if file_hash(path)!=transitions[path.name][0]:raise Refusal('Bridge source changed during transaction')
                if data is None:
                    if path.exists():path.unlink()
                else:self._write(path,data)
        except BaseException:
            # Preserve journal on any interruption. No implicit resend/recovery.
            raise
        self._clear_lock()

    def stage(self,apply=False):
        plan=self.plan()
        if plan['installed']:raise Refusal('Bridge already staged')
        if not apply:return plan
        self.gate();self.plan()
        backup=self.checked(self.state/'agent-before.py')
        if backup.exists():raise Refusal('Preexisting bridge backup; preserve and reconcile')
        self.plan()
        writes=[(backup,self.target('exec_agent.py').read_bytes())]
        writes.extend((self.target(n),(self.candidate/n).read_bytes()) for n in NAMES)
        writes.append((self.receipt,self.receipt_bytes()));self._transaction(writes,'stage')
        return dict(command='stage',applied=True,files=2,worker_starts=0,broker_calls=0)

    def rollback(self,apply=False):
        self.plan()
        receipt=json.loads(self.receipt.read_text());backup=self.checked(self.state/'agent-before.py')
        if file_hash(backup)!=receipt['before']['exec_agent.py']:raise Refusal('Bridge rollback preimage differs')
        if not apply:return dict(command='rollback',writes=0,planned_files=list(NAMES),worker_starts=0,broker_calls=0)
        self.gate();self.plan()
        self._transaction([(self.target('exec_agent.py'),backup.read_bytes()),
                           (self.target('execution_connection.py'),None),(self.receipt,None)],'rollback')
        return dict(command='rollback',applied=True,files=2,worker_starts=0,broker_calls=0)

    def recover(self,apply=False):
        journal=json.loads(self.checked(self.lock/'transaction.json').read_text())
        if journal.get('id')!=ID:raise Refusal('Unknown bridge recovery journal')
        transitions=self.transitions(journal.get('operation'))
        if len(journal['rows'])!=len(transitions) or {r['path'] for r in journal['rows']}!=set(transitions):
            raise Refusal('Incomplete/duplicate bridge recovery rows')
        writes=[]
        for i,row in enumerate(journal['rows']):
            path=self.row_target(row['path'])
            if (row['before_sha256'],row['after_sha256'])!=transitions[row['path']]:raise Refusal('Unreviewed recovery transition')
            if row['backup']!=f'preimage-{i}.bin' or file_hash(path) not in {row['before_sha256'],row['after_sha256']}:
                raise Refusal('Unknown bridge recovery drift')
            data=None
            if row['before_sha256'] is not None:
                backup=self.checked(self.lock/row['backup'])
                if file_hash(backup)!=row['before_sha256']:raise Refusal('Recovery preimage changed')
                data=backup.read_bytes()
            writes.append((path,data))
        if not apply:return dict(command='recover',writes=0,planned_files=len(writes),worker_starts=0,broker_calls=0)
        self.gate()
        # Gate execution can take time. Revalidate the entire pinned set and
        # rollback preimages before any recovery mutation.
        for path,data in writes:
            self.checked(path)
            if file_hash(path) not in set(transitions[path.name]):raise Refusal('Bridge changed after recovery gate')
            before_sha=transitions[path.name][0]
            if (hashlib.sha256(data).hexdigest() if data is not None else None)!=before_sha:
                raise Refusal('Bridge recovery preimage changed after gate')
        for path,data in reversed(writes):
            self.checked(path)
            if file_hash(path) not in set(transitions[path.name]):raise Refusal('Bridge changed after recovery gate')
            if data is None:
                if path.exists():path.unlink()
            else:atomic_write(path,data)
        self._clear_lock()
        return dict(command='recover',applied=True,worker_starts=0,broker_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('preflight','stage','rollback','recover'))
    p.add_argument('--apply',action='store_true');args=p.parse_args();d=BridgeDeployment()
    result=d.plan() if args.command=='preflight' else getattr(d,args.command)(args.apply)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
