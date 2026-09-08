"""Build a narrow, hash-verified candidate; no broker imports or installation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from broker_runtime.prepare import change_function, replace_once

HERE = Path(__file__).resolve().parent


def function_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def patch_executor(source):
    for name, adding in (("_do_close_resize", False), ("_do_add_to_position", True)):
        original = function_source(source, name)
        legacy = original.replace("def " + name + "(", "def _legacy" + name + "(", 1)
        replacement = f'''{legacy}

def {name}(ib, p, acct, host, port, main_cid):
    if acct != "primary":
        return _legacy{name}(ib, p, acct, host, port, main_cid)
    import position_actions
    return position_actions.run(globals(), ib, p, acct, host, port, main_cid, adding={adding})
'''
        source = change_function(source, name, lambda _, body=replacement: body)
    prepare = function_source(source, "_prepare_fast_position")
    prepare = prepare.replace("def _prepare_fast_position(", "def _prepare_position_action_add(", 1)
    prepare = prepare.replace('    pos, err = _exact_position(ib, p)', '    import position_actions\n    pos, err = _exact_position(ib, p)', 1)
    prepare = replace_once(prepare, "_validate_exit_topology(legs, held)", "position_actions.validate_topology(legs)")
    prepare = replace_once(prepare, "_, err = _scaled_exit_legs(legs, total)", "err = position_actions.validate_total(legs, total)")
    prepare = replace_once(prepare, "totals = [held + qty]", "totals = [qty]")
    source = change_function(source, "_prepare_fast_position", lambda old: old + "\n\n" + prepare)
    source = change_function(source, "main", lambda old: replace_once(
        old, "if t in DISABLED_UNSAFE_MUTATIONS:",
        'if t in DISABLED_UNSAFE_MUTATIONS and not (acct == "primary" and t == "add_to_position"):'))
    return source


def patch_agent(source):
    def validate(old):
        return replace_once(old, "    reasons = []", '''    import position_action_agent
    if position_action_agent.applies(cmd):
        return position_action_agent.validate(cmd)
    reasons = []''')
    source = change_function(source, "_validate", validate)
    source = change_function(source, "_preview", lambda old: replace_once(
        old, '    t, p, acct = cmd.get("type"), (cmd.get("payload") or {}), cmd.get("account")',
        '''    import position_action_agent
    if position_action_agent.applies(cmd):
        return position_action_agent.preview(cmd)
    t, p, acct = cmd.get("type"), (cmd.get("payload") or {}), cmd.get("account")'''))
    source = change_function(source, "_live_eligible", lambda old: replace_once(
        old, "if armed_type in DISABLED_UNSAFE_MUTATIONS:",
        'if armed_type in DISABLED_UNSAFE_MUTATIONS and not (cmd.get("account") == "primary" and armed_type == "add_to_position"):'))
    # Serialize subprocesses from user commands and background reconciliation.
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_live")
    original = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    unlocked = original.replace("async def _execute_live(", "async def _execute_live_unlocked(", 1)
    wrapper = '''

_POSITION_EXEC_LOCK = None

async def _execute_live(cmd):
    global _POSITION_EXEC_LOCK
    if _POSITION_EXEC_LOCK is None:
        _POSITION_EXEC_LOCK = asyncio.Lock()
    async with _POSITION_EXEC_LOCK:
        return await _execute_live_unlocked(cmd)
'''
    source = change_function(source, "_execute_live", lambda _: unlocked + wrapper)
    def run_once(old):
        old = replace_once(old, "        so = asyncio.create_task(_scheduled_option_loop(ws))",
                           "        so = asyncio.create_task(_scheduled_option_loop(ws))\n"
                           "        import position_action_agent\n"
                           "        pa = asyncio.create_task(position_action_agent.loop(globals(), ws))")
        old = replace_once(old, "            so.cancel()", "            so.cancel()\n            pa.cancel()")
        return replace_once(old, "asyncio.gather(hb, bk, so,", "asyncio.gather(hb, bk, so, pa,")
    return change_function(source, "_run_once", run_once)


def prepare(source_root, output):
    if output.exists():
        raise ValueError("candidate directory must be new")
    expected = json.loads((HERE / "position_action_source_hashes.json").read_text())
    rendered = {}
    for name, patch in (("execute_order.py", patch_executor), ("exec_agent.py", patch_agent)):
        raw = (source_root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError(f"reviewed source changed: {name}")
        rendered[name] = patch(raw.decode("utf-8-sig").replace("\r\n", "\n"))
    for name in ("position_actions.py", "position_action_agent.py", "execution_lifecycle.py"):
        rendered[name] = (HERE / name).read_text(encoding="utf-8")
    for name, text in rendered.items():
        compile(text, name, "exec")
    output.mkdir(parents=True)
    for name, text in rendered.items():
        (output / name).write_text(text, encoding="utf-8")
    manifest = {"source": expected, "candidate": {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in rendered}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
