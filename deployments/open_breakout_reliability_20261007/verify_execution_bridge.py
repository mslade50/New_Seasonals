"""Prepare and test the reconnect helper offline; never imports the live agent."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent

def main():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT));import offline_guard;offline_guard.install()
    source=ROOT/'execution-bridge/source';sys.path.insert(0,str(source))
    from broker_runtime.prepare_execution_connection import prepare
    candidate=ROOT/'execution-bridge/candidate'
    if not candidate.exists():
        prepare(Path(r'C:\Users\McKinley Slade\OneDrive\trading_ibkr'),candidate)
    import pytest
    output=ROOT.parent/'reliability-bridge-evidence';output.mkdir(exist_ok=True)
    os.chdir(source);stream=io.StringIO()
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=pytest.main(['tests/test_execution_connection.py','-q','--tb=short','-p','no:cacheprovider',
                           '--basetemp='+str(output/'fixtures'),'--junitxml='+str(output/'bridge.xml')])
    (output/'bridge.txt').write_text(stream.getvalue())
    cases=list(ET.parse(output/'bridge.xml').iter('testcase'))
    counts=dict(tests=len(cases),failed=sum(c.find('failure') is not None for c in cases),
                errors=sum(c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases))
    counts['passed']=counts['tests']-counts['failed']-counts['errors']-counts['skipped']
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),counts=counts,
                source_commits=['5d9b551d0825e83a284082313af29a7714902fab','73eb77e3036b8255afd5a7925cf02579e8ddd84b'],
                native_broker_network_attempts=offline_guard.ATTEMPTS,installed=False,restarted=False,
                credentials_read=False,scheduler_queried=False,relay_deployed=False)
    (output/'bridge.json').write_text(json.dumps(report,indent=2)+'\n')
    print(stream.getvalue()[-4500:]);print(json.dumps(report,indent=2))
    return int(result) or int(bool(offline_guard.ATTEMPTS))

if __name__=='__main__':raise SystemExit(main())
