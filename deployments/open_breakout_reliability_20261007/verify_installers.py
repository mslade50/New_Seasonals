"""Local guarded installer suite; applies only to injected disposable roots."""
from pathlib import Path
from datetime import datetime,timezone
import contextlib
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent

def main():
    sys.dont_write_bytecode=True
    output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    import offline_guard
    offline_guard.install()
    import pytest
    stream=io.StringIO();started=datetime.now(timezone.utc).isoformat()
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=pytest.main(['test_deployment.py','test_reliability_deployment.py','-q','--tb=short','-p','no:cacheprovider',
                           '--basetemp='+str(output/'fixtures'),'--junitxml='+str(output/'installers.xml')])
    (output/'installers.txt').write_text(stream.getvalue())
    cases=list(ET.parse(output/'installers.xml').iter('testcase'))
    counts={k:sum(c.find(tag) is not None for c in cases) for k,tag in (('failed','failure'),('errors','error'),('skipped','skipped'))}
    counts.update(tests=len(cases),passed=len(cases)-sum(counts.values()))
    report=dict(started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),counts=counts,
                native_broker_network_attempts=offline_guard.ATTEMPTS,production_applied=False,
                production_workers_started=False,scheduler_modified=False,inert_installer_roots=True)
    (output/'installers.json').write_text(json.dumps(report,indent=2)+'\n')
    print(stream.getvalue()[-6000:]);print(json.dumps(report,indent=2))
    return int(result) or int(bool(offline_guard.ATTEMPTS))

if __name__=='__main__':raise SystemExit(main())
