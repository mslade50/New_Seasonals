"""Prepare only an agent connection candidate; never installs, starts or reads credentials."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

from broker_runtime.prepare import change_function, replace_once

REVIEWED_AGENT_SHA256 = '7ec1f52ca49ea66de4504996e4889536fef5494e4ea490ca5312f8b5490bd817'
HERE = Path(__file__).resolve().parent


def patch(source):
    original_tree = ast.parse(source)
    def session(text):
        text = replace_once(text, 'async def _run_once() -> None:',
                            'async def _run_once(connection_state) -> None:\n    import execution_connection')
        text = replace_once(text, '        log(f"connected -> {WS_URL}")',
                            '        monitor = execution_connection.Session(connection_state, log, secrets=execution_connection.secrets_from(globals()))')
        text = replace_once(text, '                await asyncio.sleep(HEARTBEAT_S)',
                            '                due = time.monotonic() + HEARTBEAT_S\n                await asyncio.sleep(HEARTBEAT_S)\n                monitor.loop_wake(due)')
        text = replace_once(text, '                await ws.send(json.dumps({"type": "heartbeat", "t": time.time()}))',
                            '                await ws.send(json.dumps(monitor.heartbeat()))\n                monitor.tick()')
        text = replace_once(text, '        try:\n            async for raw in ws:',
                            '        async def receive():\n            async for raw in ws:')
        text = replace_once(text, '                if msg.get("type") == "command":',
                            '                monitor.observe(msg)\n                monitor.busy = msg.get("type") != "ack"\n                if msg.get("type") == "command":')
        text = replace_once(text, '        finally:\n            hb.cancel()',
                            '                monitor.busy = False\n        receiver = asyncio.create_task(receive())\n        try:\n            await execution_connection.supervise(receiver, hb, ws.close)\n        finally:\n            monitor.finish(ws)\n            receiver.cancel()\n            hb.cancel()')
        text = replace_once(text, 'await asyncio.gather(hb, bk, so, pa, return_exceptions=True)',
                            'await asyncio.gather(receiver, hb, bk, so, pa, return_exceptions=True)')
        return text
    source = change_function(source, '_run_once', session)
    source = change_function(source, 'main', lambda _: '''async def main() -> None:
    import execution_connection
    await execution_connection.run(globals(), _run_once)
''')
    # Correct the obsolete explanation without changing protocol settings in this release.
    def connection_doc(text):
        function = ast.parse(text).body[0]
        doc = function.body[0]
        if not isinstance(doc, ast.Expr) or not isinstance(doc.value, ast.Constant) or not isinstance(doc.value.value, str):
            raise ValueError('Missing reviewed connection docstring')
        lines = text.splitlines(keepends=True)
        replacement = '    """Protocol pings remain disabled for compatibility in this release.\n' \
            '    Cloudflare currently supports automatic pong responses. Application\n' \
            '    heartbeats retain the existing cadence; connection diagnostics track\n' \
            '    ACKs and supervise heartbeat errors without replaying commands.\n    """\n'
        return ''.join(lines[:doc.lineno-1]) + replacement + ''.join(lines[doc.end_lineno:])
    source = change_function(source, '_connect', connection_doc)
    updated_tree = ast.parse(source)
    def unchanged_nodes(tree):
        return [ast.dump(node) for node in tree.body if getattr(node, 'name', '') not in {'_connect', '_run_once', 'main'}]
    if unchanged_nodes(original_tree) != unchanged_nodes(updated_tree):
        raise ValueError('Unexpected change outside connection functions')
    return source


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents or output.exists():
        raise ValueError('Output must be a new directory separate from the runtime.')
    path = source/'exec_agent.py'
    original = path.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != REVIEWED_AGENT_SHA256:
        raise ValueError('Agent source changed; re-review connection patch before promotion.')
    updated = patch(original.decode('utf-8-sig').replace('\r\n', '\n'))
    output.mkdir(parents=True)
    (output/'exec_agent.py').write_text(updated, encoding='utf-8')
    shutil.copyfile(HERE/'execution_connection.py', output/'execution_connection.py')
    manifest = dict(source=str(source), source_agent_sha256=actual, installed=False, restarted=False,
                    files={name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                           for name in ('exec_agent.py', 'execution_connection.py')})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
