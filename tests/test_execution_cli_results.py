"""Run actual installed CLI boundaries with a simulated broker in a child process."""
import json
import asyncio
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.spent_preparers import retired_execution_repairs

SOURCE = Path(os.environ.get("IBKR_REVIEW_SOURCE", "C:/Users/McKinley Slade/OneDrive/trading_ibkr"))
REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/execution_runtime"


def run_async(coroutine):
    # asyncio.run clears the thread's default loop, breaking later ib_insync
    # shape imports. Keep this reporting test's event loop isolated.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


CHILD = r'''
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace as N
from broker_runtime import order_mutations, prepare_execution_repairs
from tests.test_execution_repair_dispatch import edit_fixture
source, state, operation, scenario = sys.argv[1:]
broker, ns, payload = edit_fixture(Path(state))
payload.update(new_qty=60)
if scenario == 'rejected': payload['new_qty'] = -1
if scenario == 'unknown': broker.fail_resize = True
class Event:
    def __iadd__(self, fn): return self
broker.connect = lambda *a, **k: None
broker.disconnect = lambda: None
broker.errorEvent = Event()
text = Path(source).joinpath('execute_order.py').read_text(encoding='utf-8-sig')
if 'import order_mutations' not in text:
    text = prepare_execution_repairs.patch_executor(text)
tree = ast.parse(text)
names = {'_out','_do_modify','_do_cancel','main'}
nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names
         or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
             t.id in {'SUPPORTED','DISABLED_UNSAFE_MUTATIONS'} for t in n.targets)]
cmd = dict(id=payload['_command_id'],type=operation,account='primary',payload=payload)
ns.update(json=json,sys=N(argv=['fixture',json.dumps(cmd)]),
          LIVE_ENABLED=True,LIVE_ACCOUNTS={'primary'},LIVE_TYPES={operation},
          PORTS={'primary':('',0,7)},IB=lambda:broker,
          _resolve_broker_account=lambda *a:'PRIMARY',_on_err=lambda *a:None)
sys.modules['order_mutations'] = order_mutations
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-cli','exec'),ns)
result = ns['main']()
Path(state).joinpath('simulation.json').write_text(json.dumps(broker.mutations))
sys.exit(result)
'''


def invoke(tmp_path, operation, scenario):
    source = SOURCE if (SOURCE / "execute_order.py").exists() else FIXTURE
    process = subprocess.run([sys.executable, "-c", CHILD, str(source), str(tmp_path),
                              operation, scenario], cwd=REPO, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    # Agent parses the complete stdout, so a second JSON document is a failure.
    result = json.loads(process.stdout)
    assert type(result["ok"]) is bool
    assert (result["state"] == "executed") == result["ok"]
    assert result["detail"]
    return result, json.loads((tmp_path / "simulation.json").read_text())


@retired_execution_repairs
def test_reviewed_cli_fixture_matches_installed_boundaries():
    from broker_runtime.prepare_execution_repairs import patch_executor
    path = SOURCE / "execute_order.py"
    if not path.exists():
        pytest.skip("Runtime unavailable; portable CLI fixture still runs all behavior tests")
    source = path.read_text(encoding="utf-8-sig")
    if "import order_mutations" not in source:
        source = patch_executor(source)
    live = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for node in ast.parse((FIXTURE / "execute_order.py").read_text()).body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == live[node.name]


@pytest.mark.parametrize("operation", ["modify", "cancel"])
@retired_execution_repairs
def test_actual_cli_emits_one_terminal_result_and_replay_never_resubmits(tmp_path, operation):
    result, mutations = invoke(tmp_path, operation, "executed")
    assert result["state"] == "executed" and len(mutations) == 1
    replay, replay_mutations = invoke(tmp_path, operation, "executed")
    assert replay == result and replay_mutations == []
    record = next((tmp_path / "order_edits").glob("*.json"))
    assert json.loads(record.read_text())["phase"] == "done"


@pytest.mark.parametrize("scenario", ["rejected", "unknown"])
@retired_execution_repairs
def test_actual_cli_preserves_rejection_and_transmission_uncertainty(tmp_path, scenario):
    result, mutations = invoke(tmp_path, "modify", scenario)
    assert result["state"] == scenario
    assert len(mutations) == (1 if scenario == "unknown" else 0)
    record = json.loads(next((tmp_path / "order_edits").glob("*.json")).read_text())
    assert record["phase"] == ("attention" if scenario == "unknown" else "done")


def test_completed_edit_reports_after_reconnect_without_execution(tmp_path):
    from broker_runtime import position_actions as actions
    from broker_runtime.position_action_agent import report_completed_edits
    messages = []
    class Socket:
        async def send(self, message): messages.append(json.loads(message))
    result = dict(ok=True, state="executed", detail="Confirmed by broker snapshot")
    actions.save(tmp_path, dict(version=1, id="done", phase="done", result=result))
    actions.save(tmp_path, dict(version=1, id="uncertain", phase="mutating"))
    delivered = {}
    run_async(report_completed_edits(tmp_path, Socket(), delivered))
    run_async(report_completed_edits(tmp_path, Socket(), delivered))
    assert len(messages) == 1 and messages[0]["id"] == "done"
    assert messages[0]["state"] == "executed"
    run_async(report_completed_edits(tmp_path, Socket(), {}))
    assert len(messages) == 2
    assert json.loads(actions.record_path(tmp_path, "uncertain").read_text())["phase"] == "mutating"


def test_invalid_completed_edit_is_not_reported_as_success(tmp_path):
    from broker_runtime import position_actions as actions
    from broker_runtime.position_action_agent import report_completed_edits
    class Socket:
        async def send(self, message): pytest.fail("Invalid receipt must not be emitted")
    actions.save(tmp_path, dict(version=1,id="invalid",phase="done",result=0))
    with pytest.raises(ValueError, match="invalid terminal"):
        run_async(report_completed_edits(tmp_path, Socket(), {}))
