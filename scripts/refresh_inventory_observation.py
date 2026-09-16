"""Publish one read-only Gateway observation; never start the command agent."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class InventoryRefreshError(RuntimeError):
    """Only fixed, credential-free diagnostics may cross the scan boundary."""


def _snapshot_failure(primary):
    # Never expose arbitrary broker response text. The installed snapshot emits
    # these fixed forms; unknown responses retain an explicit boundary only.
    error = str(primary.get('error') or '')
    if error.startswith('not connected ('):
        return 'Primary broker connection failed during read-only inventory query'
    if error:
        return 'Primary broker connected but its account/position/order query failed'
    return 'Primary broker execution query did not complete'


def refresh_local_inventory(base_url, *, run=subprocess.run, post=None):
    snapshot=Path(os.environ.get('INVENTORY_SNAPSHOT_PATH',''))
    token=os.environ.get('EXEC_AGENT_TOKEN','').strip()
    if not snapshot.is_file() or not token:
        raise InventoryRefreshError('local read-only inventory refresh is not configured')
    python=os.environ.get('INVENTORY_SNAPSHOT_PYTHON',sys._base_executable)
    query = Path(__file__).with_name('query_inventory_snapshot.py')
    try:
        result=run([python,str(query),str(snapshot)],cwd=str(snapshot.parent),capture_output=True,
                   text=True,encoding='utf-8',timeout=60,check=True)
    except subprocess.TimeoutExpired:
        raise InventoryRefreshError('Primary read-only inventory subprocess timed out') from None
    except subprocess.CalledProcessError:
        raise InventoryRefreshError('Primary read-only inventory subprocess failed') from None
    book=json.loads(result.stdout)
    primary=[a for a in book.get('accounts',[]) if a.get('key')=='primary']
    if len(primary)!=1 or primary[0].get('error') or primary[0].get('fills_complete') is not True:
        raise InventoryRefreshError(_snapshot_failure(primary[0]) if len(primary)==1
                                    else 'Primary read-only query returned ambiguous accounts')
    book['at']=int(time.time()*1000)
    if post is None:
        import requests
        post=requests.post
    reply=post(base_url.rstrip('/')+'/inventory-observation',json=book,
               headers={'Authorization':'Bearer '+token},timeout=30)
    reply.raise_for_status()
    if reply.json().get('ok') is not True:
        raise InventoryRefreshError('inventory observation was not accepted by the relay')

