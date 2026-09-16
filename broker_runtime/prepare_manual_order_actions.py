"""Prepare a source-pinned manual-order candidate without activating it."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from broker_runtime.prepare import change_function, replace_once

HERE = Path(__file__).resolve().parent


def patch_executor(source):
    for name, modify, tail in (("_do_cancel", False, ""), ("_do_modify", True, ", acct")):
        source = change_function(source, name, lambda _, name=name, modify=modify, tail=tail:
            f"def {name}(ib, p, host, port, main_cid{tail}):\n"
            f"    import manual_order_actions\n"
            f"    return manual_order_actions.run(globals(), ib, p, host, port, main_cid, modify={modify})\n")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SUPPORTED" for t in n.targets))
    anchor = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    source = replace_once(source, anchor, anchor + '\nSUPPORTED.add("reconcile_exits")')
    return change_function(source, "main", lambda text: replace_once(text,
        '        if t == "cancel":',
        '        if t == "reconcile_exits":\n'
        '            import reconcile_position_exits\n'
        '            return reconcile_position_exits.run(globals(), ib, p, host, port, cid)\n'
        '        if t == "cancel":'))


def patch_agent(source):
    def validate(text):
        anchor = '    t, p, acct = cmd.get("type"), (cmd.get("payload") or {}), cmd.get("account")'
        return replace_once(text, anchor, anchor + '\n    if t in {"cancel", "modify"}:\n'
                            '        import manual_order_actions\n'
                            '        return manual_order_actions.validate(cmd)\n'
                            '    if t == "reconcile_exits":\n'
                            '        import reconcile_position_exits\n'
                            '        return reconcile_position_exits.validate(cmd)')
    source = change_function(source, "_validate", validate)
    source = change_function(source, "_preview", lambda text: replace_once(text,
        '    if t == "cancel":',
        '    if t == "reconcile_exits":\n'
        '        return {"summary": "Reconcile existing exits to live position",\n'
        '                "legs": ["Proportional exit quantities; prices and schedules preserved. No new close or re-add."]}\n'
        '    if t == "cancel":'))
    return source


def prepare(source, output):
    expected = json.loads((HERE / "manual_order_source_hashes.json").read_text())
    for name, digest in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"reviewed runtime source changed: {name}")
    if output.exists():
        raise ValueError("candidate directory must be new")
    rendered = {name: (HERE / name).read_text(encoding="utf-8")
                for name in ("manual_order_actions.py", "reconcile_position_exits.py")}
    for name, patch in (("execute_order.py", patch_executor), ("exec_agent.py", patch_agent)):
        rendered[name] = patch((source / name).read_text(encoding="utf-8-sig"))
    for name, text in rendered.items():
        compile(text, name, "exec")
    output.mkdir(parents=True)
    for name, text in rendered.items():
        (output / name).write_text(text, encoding="utf-8")
    manifest = dict(source=expected, candidate={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in rendered})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
