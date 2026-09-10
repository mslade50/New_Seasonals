"""Publish one read-only Gateway observation; never start the command agent."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def refresh_local_inventory(base_url, *, run=subprocess.run, post=None):
    snapshot=Path(os.environ.get('INVENTORY_SNAPSHOT_PATH',''))
    token=os.environ.get('EXEC_AGENT_TOKEN','').strip()
    if not snapshot.is_file() or not token:
        raise RuntimeError('local read-only inventory refresh is not configured')
    python=os.environ.get('INVENTORY_SNAPSHOT_PYTHON',sys._base_executable)
    result=run([python,str(snapshot)],cwd=str(snapshot.parent),capture_output=True,
               text=True,encoding='utf-8',timeout=60,check=True)
    book=json.loads(result.stdout)
    primary=[a for a in book.get('accounts',[]) if a.get('key')=='primary']
    if len(primary)!=1 or primary[0].get('error') or primary[0].get('fills_complete') is not True:
        raise RuntimeError('Primary read-only query failed')
    book['at']=int(time.time()*1000)
    if post is None:
        import requests
        post=requests.post
    reply=post(base_url.rstrip('/')+'/inventory-observation',json=book,
               headers={'Authorization':'Bearer '+token},timeout=30)
    reply.raise_for_status()
    if reply.json().get('ok') is not True:
        raise RuntimeError('inventory observation was not accepted')

