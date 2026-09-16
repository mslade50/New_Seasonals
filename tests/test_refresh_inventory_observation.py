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
        assert command[0]=='fixture-python' and command[2]==str(snapshot) and kwargs['cwd']==str(tmp_path)
        assert command[1].endswith('query_inventory_snapshot.py')
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


@pytest.mark.parametrize('error,expected', [
    ('not connected (TimeoutError)', 'broker connection failed'),
    ('snapshot error (RuntimeError)', 'account/position/order query failed'),
    (None, 'execution query did not complete'),
])
def test_refresh_reports_safe_failure_boundary(tmp_path,monkeypatch,error,expected):
    snapshot=tmp_path/'book_snapshot.py';snapshot.write_text('# fixture')
    monkeypatch.setenv('INVENTORY_SNAPSHOT_PATH',str(snapshot))
    monkeypatch.setenv('EXEC_AGENT_TOKEN','secret-fixture')
    def run(*args,**kwargs):
        return SimpleNamespace(stdout=json.dumps({'accounts':[{'key':'primary','error':error,'fills_complete':False}]}))
    with pytest.raises(RuntimeError,match=expected):
        refresh_local_inventory('https://fixture.invalid',run=run,post=lambda *a,**k:pytest.fail('published failed query'))


def test_query_uses_primary_only_and_separate_client(tmp_path):
    from scripts.query_inventory_snapshot import query
    path=tmp_path/'snapshot.py'
    path.write_text("ACCOUNTS=[dict(key='primary',cid=122),dict(key='pa',cid=146)]\ndef snap_account(account):\n    assert account['key']=='primary' and account['cid']==8122\n    return account\n")
    assert query(path)=={'accounts':[{'key':'primary','cid':8122}]}
