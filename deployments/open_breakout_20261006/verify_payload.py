"""Run the aggregate suite against installed payload bytes in disposable roots only."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def run_staged(repo, output):
    sys.dont_write_bytecode = True
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    for key in ('INTRADAY_COORDINATION_ENABLED', 'INTRADAY_COORDINATION_DB', 'OPEN_BREAKOUT_LIVE_ACK'):
        os.environ.pop(key, None)
    import offline_guard
    offline_guard.install()
    import pytest
    stream = io.StringIO()
    started = datetime.now(timezone.utc).isoformat()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        paths=['tests/'+p.name for p in sorted((ROOT/'offline_tests').glob('test_*.py'))]
        result = pytest.main(paths+['-q', '--disable-warnings', '--tb=short', '-p', 'no:cacheprovider',
                              '--basetemp='+str(output/'fixtures'), '--junitxml='+str(output/'aggregate.xml')])
    (output/'aggregate.txt').write_text(stream.getvalue(), encoding='utf-8')
    cases = list(ET.parse(output/'aggregate.xml').iter('testcase'))
    counts = dict(tests=len(cases),
                  passed=sum(not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases),
                  failed=sum(c.find('failure') is not None for c in cases),
                  errors=sum(c.find('error') is not None for c in cases),
                  skipped=sum(c.find('skipped') is not None for c in cases))
    imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if name.startswith(('open_breakout', 'intraday_coordination', 'legend.')) and getattr(module, '__file__', None)}
    assert imports and all(Path(path).is_relative_to(repo) for path in imports.values()), imports
    report = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  exit_code=int(result), counts=counts, native_broker_network_attempts=offline_guard.ATTEMPTS,
                  imports=imports, actual_install_payload=True, native_close_qualified=False,
                  production_applied=False, production_workers_started=False, scheduler_modified=False)
    (output/'aggregate.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(stream.getvalue()[-6000:] if result else stream.getvalue().splitlines()[0:8])
    print(json.dumps({k:v for k,v in report.items() if k!='imports'}, indent=2))
    return int(result) or int(bool(offline_guard.ATTEMPTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-staged', nargs=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.run_staged:
        return run_staged(*[Path(p).resolve() for p in args.run_staged])
    import installer
    import test_deployment
    deployment = installer.Deployment()
    deployment.report()  # Fail on any production/candidate drift before copying fixture bytes.
    before = {r['root']+':'+r['path']: installer.file_hash(deployment.target(r['root'], r['path']))
              for r in deployment.manifest['prerequisites']}
    output = (args.output or deployment.roots['repo']/'artifacts/open_breakout_deployment').resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='payload-', dir=output))
    d, roots, package = test_deployment.fixture.__wrapped__(work)
    d.stage(True)  # Mapped disposable roots, fake process inventory and fake clock only.
    repo = roots['repo']
    # The unchanged production test is a baseline prerequisite; run it once through its reviewed copy.
    shutil.copytree(ROOT/'offline_tests', repo/'tests', dirs_exist_ok=True)
    for path in repo.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    # Child replay/status CLI calls inherit the same native-path-denied guard.
    (repo/'sitecustomize.py').write_text('from offline_guard import install\ninstall()\n', encoding='utf-8')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join((str(repo), str(ROOT))))
    env['INTRADAY_COORDINATION_ENABLED']='0'
    env.pop('INTRADAY_COORDINATION_DB', None)
    result = subprocess.run([sys.executable, '-B', str(__file__), '--run-staged', str(repo), str(output)],
                            cwd=repo, env=env, text=True, capture_output=True, timeout=240)
    (output/'aggregate-console.txt').write_text(result.stdout+result.stderr, encoding='utf-8')
    print(result.stdout+result.stderr)
    after = {r['root']+':'+r['path']: installer.file_hash(deployment.target(r['root'], r['path']))
             for r in deployment.manifest['prerequisites']}
    report = json.loads((output/'aggregate.json').read_text())
    report['production_hashes_unchanged'] = before == after
    report['production_prerequisite_count'] = len(before)
    report['payload_hashes'] = d.installed()[0]['installed_hashes']
    (output/'aggregate.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result.returncode or int(before != after)


if __name__ == '__main__':
    raise SystemExit(main())
