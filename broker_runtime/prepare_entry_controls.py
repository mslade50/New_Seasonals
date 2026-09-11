"""Prepare narrowly scoped execution changes; never install, arm, or connect."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

NOTIONAL_SETTING = "LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS"
NOTIONAL_HELPER = '''def _futures_notional_exempt(acct):
    """Notional-only exemption; quantity and risk checks use their own policy."""
    accounts = {value.strip().lower() for value in
                os.environ.get("LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS", "").split(",")
                if value.strip()}
    return _uncapped_futures(acct) or str(acct or "").lower() in accounts


'''
GATE = 'if sec_type != "CASH" and not (sec_type == "FUT" and _uncapped_futures(acct)):'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence: {old[:90]}")
    return source.replace(old, new, 1)


def patch(source, *, executor):
    source = replace_once(source, "def _uncapped_options(acct):",
                          NOTIONAL_HELPER + "def _uncapped_options(acct):")
    source = replace_once(source, GATE, GATE.replace("_uncapped_futures", "_futures_notional_exempt"))
    if executor:
        source = replace_once(
            source,
            'SUPPORTED = {"entry_bracket", "close_only", "close_resize", "flatten",\n'
            '             "option_spread", "exit_attach"}',
            'SUPPORTED = {"entry_bracket", "close_only", "close_resize", "flatten",\n'
            '             "option_spread", "exit_attach", "add_to_position"}',
        )
    ast.parse(source)
    return source


def prepare(source_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "prepared_only", "files": {},
                "proposed_environment": {NOTIONAL_SETTING: "pa"}}
    for name in ("exec_agent.py", "execute_order.py"):
        original = (Path(source_dir) / name).read_bytes()
        candidate = patch(original.decode("utf-8-sig").replace("\r\n", "\n"),
                          executor=name == "execute_order.py").encode("utf-8")
        (output / name).write_bytes(candidate)
        (output / (name + ".original")).write_bytes(original)
        manifest["files"][name] = {
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
