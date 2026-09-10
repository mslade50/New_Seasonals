import json
from types import SimpleNamespace
import pytest
from scripts.refresh_inventory_observation import refresh_local_inventory


def test_refresh_runs_only_snapshot_and_publishes_observation(tmp_path,monkeypatch):
    snapshot=tmp_path/'book_snapshot.py';snapshot.write_text('# read-only fixture')
    monkeypatch.setenv('INVENTORY_SNAPSHOT_PATH',str(snapshot))
    monkeypatch.setenv('INVENTORY_SNAPSHOT_PYTHON','fixture-python')
    monkeypatch.setenv('EXEC_AGENT_TOKEN','fixture-token')
    calls=[]
    def run(command,**kwargs):
        assert command==['fixture-python',str(snapshot)] and kwargs['cwd']==str(tmp_path)
        assert kwargs['check'] and kwargs['timeout']==60
        return SimpleNamespace(stdout=json.dumps({'accounts':[{'key':'primary','fills_complete':True}]}))
    def post(url,**kwargs):
        calls.append(url)
        assert kwargs['headers']=={'Authorization':'Bearer fixture-token'}
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'ok':True})
    refresh_local_inventory('https://fixture.invalid/',run=run,post=post)
    assert calls==['https://fixture.invalid/inventory-observation']


def test_unconfigured_refresh_cannot_start_any_process(monkeypatch):
    monkeypatch.delenv('INVENTORY_SNAPSHOT_PATH',raising=False)
    with pytest.raises(RuntimeError,match='not configured'):
        refresh_local_inventory('https://fixture.invalid',run=lambda *a,**k:pytest.fail('process started'))
