"""Pinned local startup settings. No broker imports, connection, orders or worker starts."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

def configure(worker, *, argv=None, now=None):
    root = Path(__file__).resolve().parents[1]
    # Any inherited enable flag/DB is deliberately replaced by the reviewed local settings.
    os.environ['INTRADAY_COORDINATION_ENABLED'] = '0'
    os.environ.pop('INTRADAY_COORDINATION_DB', None)
    args = list(sys.argv[1:] if argv is None else argv)
    # Existing safety/verification commands must remain available without the entry fence.
    live = ((worker == 'legend' and not any(a in args for a in ('--kill','--verify-only','--dry-run')))
            or (worker == 'open_breakout' and args[:1] == ['live-session']))
    if worker not in {'legend', 'open_breakout'}:
        raise RuntimeError('Unknown coordination worker')
    if not live:
        return dict(enabled=False, worker=worker)
    runs = root/'artifacts/open_breakout_runs'
    if (runs/'combined-deployment-install.lock').exists():
        raise RuntimeError('Installation/recovery in progress; refusing live startup')
    body = json.loads((runs/'combined-deployment-settings.json').read_text(encoding='utf-8'))
    receipt = json.loads((runs/'combined-install-receipt.json').read_text(encoding='utf-8'))
    canonical = (json.dumps(body, indent=2, sort_keys=True)+'\n').encode()
    if (body.get('schema') != 1 or body.get('bundle_id') != 'open-breakout-reliability-20261007-v2'
            or type(body.get('enabled')) is not bool or Path(body.get('repo_root','')).resolve() != root
            or receipt.get('schema') != 1 or receipt.get('bundle_id') != body['bundle_id']
            or hashlib.sha256(canonical).hexdigest() != receipt.get('settings_sha256')):
        raise RuntimeError('Unknown/changed deployment settings or receipt; refusing startup')
    if not body['enabled'] or not live:
        return dict(enabled=False, worker=worker)
    day = (now or datetime.now(ZoneInfo('America/New_York'))).date().isoformat()
    try:
        effective = datetime.strptime(body['effective_from'], '%Y-%m-%d').date().isoformat()
    except (KeyError, TypeError, ValueError):
        raise RuntimeError('Invalid deployment activation session')
    if day < effective:
        raise RuntimeError('Deployment is for a future session; refusing early live startup')
    if worker == 'open_breakout':
        if '--session' not in args or args.index('--session')+1 >= len(args) or args[args.index('--session')+1] != day:
            raise RuntimeError('Live launch must name the current New York session')
    rows=body.get('runtime_hashes')
    if not isinstance(rows,list) or not rows or len({r['path'] for r in rows})!=len(rows):
        raise RuntimeError('Incomplete/duplicate reviewed runtime inventory')
    installed=receipt.get('installed_hashes',{})
    local_paths={str(root/key.split(':',1)[1]):value for key,value in installed.items() if key.startswith('repo:')}
    indexed={r['path']:r['sha256'] for r in rows}
    if not local_paths or any(indexed.get(p)!=v for p,v in local_paths.items()):
        raise RuntimeError('Reviewed runtime inventory differs from receipt')
    for row in rows:
        target = Path(row['path'])
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Reviewed runtime source changed; refusing startup: '+str(target))
    db = Path(body['coordination_db'])
    if db != root/'artifacts/open_breakout_runs/coordination/combined-ledger.sqlite':
        raise RuntimeError('Unknown shared coordination ledger path')
    if any(type(body.get(k)) is not bool for k in ('price_enabled','order_reliability_enabled','coordination_enabled','native_automatic_close_enabled')):
        raise RuntimeError('Incomplete independent rollout flags')
    if body['native_automatic_close_enabled']:
        raise RuntimeError('Automatic close is not qualified')
    os.environ['INTRADAY_COORDINATION_ENABLED'] = '1' if body['coordination_enabled'] else '0'
    if body['coordination_enabled']:
        os.environ['INTRADAY_COORDINATION_DB'] = str(db)
    return dict(enabled=True, worker=worker, coordination_enabled=body['coordination_enabled'], db=str(db))
