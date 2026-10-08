"""Refresh isolated candidate hashes. Never writes to a production root."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parent
ID='open-breakout-reliability-20261007-v2'

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    m=json.loads((ROOT/'deployment-manifest.json').read_text())
    m['bundle_id']=ID
    if 'source_candidate_zip_sha256' in m:
        m['input_combined_candidate_zip_sha256']=m.pop('source_candidate_zip_sha256')
    m['created_utc']=json.loads((ROOT/'reliability-input.json').read_text())['created_utc']
    m['reliability_qualified']=False
    m['coordination_default_enabled']=False
    m['changes']=[r for r in m['changes'] if r['path']!='open_breakout/order_reliability.py']
    for name in ('open_breakout/order_reliability.py','open_breakout/order_policy.py','open_breakout/standby.py'):
        if not any(r['path']==name for r in m['changes']):
            old=next((r for r in m['prerequisites'] if r['path']==name),None)
            m['changes'].append(dict(root='repo',path=name,before_sha256=old['sha256'] if old else None))
    for r in m['changes']:
        data=(ROOT/'payload'/r['root']/r['path']).read_bytes()
        if r.get('substitute_account'):
            r['payload_sha256']=sha(data);r['payload_bytes']=len(data)
            # Preserve the original reviewed local-account renderer's after hash.
            # Only account-neutral payload changes use this rendering contract.
            source=Path(m['roots']['repo'])/'artifacts/open_breakout_runs/config-20260925-live.json'
            account=json.loads(source.read_text(encoding='utf-8-sig'))['account']
            data=data.replace(b'__LOCAL_BROKER_ACCOUNT__',account.encode())
        r['after_sha256']=sha(data);r['bytes']=len(data)
    for r in m['prerequisites']:
        r['changed']=any(c['root']==r['root'] and c['path']==r['path'] for c in m['changes'])
    inventory=[dict(root=r['root'],path=r['path'],sha256=r['after_sha256']) for r in m['changes']]
    m['qualification_source_sha256']=sha(json.dumps(inventory,sort_keys=True).encode())
    (ROOT/'deployment-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    for p in (ROOT/'payload').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    print(json.dumps(dict(bundle_id=ID,targets=len(m['changes']),prerequisites=len(m['prerequisites']),
                          qualification_source_sha256=m['qualification_source_sha256'])))

if __name__=='__main__':main()
