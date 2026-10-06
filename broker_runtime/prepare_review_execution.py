"""Prepare a hash-pinned disabled candidate; NEVER install/run/connect/arm.

Source can be reviewed on the existing trading checkout. Outputs are new ignored
artifacts only. The user must separately review/promote compatible runtime files.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('reviewed source anchor changed: ' + old[:90])
    return text.replace(old, new, 1)


def change_function(text, name, transform):
    node = next(n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    lines = text.splitlines(keepends=True)
    before = ''.join(lines[node.lineno-1:node.end_lineno])
    after = transform(before)
    return ''.join(lines[:node.lineno-1]) + after + ''.join(lines[node.end_lineno:])


def patch_agent(text):
    def handler(source):
        anchor = '        if cid in _SEEN:'
        addition = '''        if cmd.get("type") == "review_execution":
            import review_execution_runtime
            reply.update(await review_execution_runtime.handle_agent(globals(), cmd))
            await ws.send(json.dumps(reply))
            return
'''
        return replace_once(source, anchor, addition + anchor)
    text = change_function(text, '_handle_command', handler)
    ast.parse(text)
    return text


def patch_executor(text):
    def entry(source):
        source = replace_once(source, '    order_ref = None\n', '    order_ref = None\n')
        tag_anchor = '    if any(not math.isfinite(v) or v <= 0 for v in\n'
        source = replace_once(source, tag_anchor, '''    if globals().get("_COMMAND_TYPE") == "review_execution":
        source_action = p.get("source_action")
        if source_action not in {"BUY", "SELL_SHORT"} or action != ("BUY" if source_action == "BUY" else "SELL"):
            return _out(False, "rejected", "review source side changed")
        order_ref = f"{sym}|{source_action}|{p['strategy']}|{p['ref_date']}"
''' + tag_anchor)
        source = replace_once(source, '    c = qualified[0]\n', '''    c = qualified[0]
    expected_con_id = p.get("_review_expected_con_id")
    if expected_con_id is not None and int(c.conId) != int(expected_con_id):
        return _out(False, "rejected", "reviewed exact contract changed before submission")
''')
        # Preview bypasses only acknowledgement, so it can REPORT the live risk.
        # Hard caps, contract qualification and price/timing gates still run.
        source = source.replace('ack = p.get("risk_ack") is True',
            'ack = p.get("risk_ack") is True or (p.get("_review_preflight_only") is True and globals().get("_COMMAND_TYPE") == "review_execution")')
        anchor = '    trades, chain_txt = [], []\n'
        addition = '''    if p.get("_review_preflight_only") is True:
        if globals().get("_COMMAND_TYPE") != "review_execution":
            return _out(False, "rejected", "review preflight context required")
        import review_execution_runtime
        context = review_execution_runtime.preflight_context(
            payload=p, broker_account=broker_account, contract_id=int(c.conId),
            quantity=qty, entry=entry, stop=stop, target=target,
            risk=guard_risk_usd, nlv=_nlv(ib, broker_account),
            stop_gat=stop_gat, time_gat=time_gat, parent_gtd=parent_gtd)
        return _out(True, "executed", "Read-only complete entry preflight",
                    fill={"review_preflight": context})
'''
        return replace_once(source, anchor, addition + anchor)
    text = change_function(text, '_do_entry_bracket', entry)
    def main(source):
        anchor = '    if not LIVE_ENABLED:\n'
        addition = '''    if t == "review_execution":
        import review_execution_runtime
        return review_execution_runtime.run_executor(globals(), cmd)

'''
        return replace_once(source, anchor, addition + anchor)
    text = change_function(text, 'main', main)
    ast.parse(text)
    return text


def prepare(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('candidate output must be separate from source runtime')
    if output.exists():
        raise ValueError('candidate directory must be new')
    hashes = json.loads((HERE/'review_execution_source_hashes.json').read_text())
    originals = {}
    for name, expected in hashes.items():
        raw = (source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('reviewed runtime source drift: ' + name)
        originals[name] = raw
    candidates = {}
    for name, transform in [('exec_agent.py', patch_agent), ('execute_order.py', patch_executor)]:
        candidates[name] = transform(originals[name].decode('utf-8-sig').replace('\r\n', '\n')).encode()
    for name in ['review_execution.py', 'review_execution_runtime.py']:
        candidates[name] = (HERE/name).read_bytes()
    for name, raw in candidates.items():
        compile(raw, name, 'exec')
    output.mkdir(parents=True)
    for name, raw in candidates.items():
        (output/name).write_bytes(raw)
    manifest = {'status':'disabled_candidate_only', 'installed':False, 'armed':False,
                'runtime_source_hashes':hashes,
                'candidate_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in candidates.items()},
                'new_flags_default':'REVIEW_EXECUTION_PREVIEW_ENABLED=0; REVIEW_EXECUTION_LIVE_ENABLED=0',
                'account_bindings':{'pitch':'primary','seasonal':None}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
