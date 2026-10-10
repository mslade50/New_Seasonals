"""Portable guarded aggregate from public fixtures; no local live roots required.

This verifies candidate runtime bytes; local installer/baseline certification is
separate and cannot be reproduced from CI's public fixture data.
"""
from pathlib import Path
import json
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parent

def main():
    sys.dont_write_bytecode=True
    output=Path(sys.argv[1] if len(sys.argv)>1 else 'offline-ci-evidence').resolve()
    output.mkdir(parents=True,exist_ok=True)
    repo=Path(tempfile.mkdtemp(prefix='guarded-reliability-',dir=output))
    shutil.copytree(ROOT/'runtime_template/open_breakout',repo/'open_breakout')
    shutil.copyfile(ROOT/'runtime_template/daily_execution_report.py',repo/'daily_execution_report.py')
    shutil.copytree(ROOT/'payload/repo',repo,dirs_exist_ok=True)
    shutil.copytree(ROOT/'fixtures/config',repo/'config')
    shutil.copytree(ROOT/'offline_tests',repo/'tests')
    shutil.copyfile(ROOT/'offline_guard.py',repo/'offline_guard.py')
    (repo/'sitecustomize.py').write_text('from offline_guard import install\ninstall()\n')
    # Unchanged baseline modules are public source fixtures; all modified modules
    # are copied exactly from the reviewed install payload.
    from verify_payload import run_staged
    result=run_staged(repo,output)
    p=output/'aggregate.json';report=json.loads(p.read_text())
    report.update(actual_install_payload=False,portable_public_fixture=True,
                  local_production_baselines_verified=False)
    p.write_text(json.dumps(report,indent=2)+'\n')
    return result

if __name__=='__main__':raise SystemExit(main())
