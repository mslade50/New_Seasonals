"""Create a fully hashed source/evidence zip. Never installs, starts or connects."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import argparse

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Account-bearing data is reconstituted locally only after base-hash verification.
    live=Path(r'C:\Users\McKinley Slade\dev\New_Seasonals\artifacts\open_breakout_runs\config-20260925-live.json')
    account=json.loads(live.read_text(encoding='utf-8-sig'))['account'].encode()
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
           and p.name!='package-integrity.json' and p.suffix not in {'.pyc','.zip'}]
    rows=[]
    for p in files:
        data=p.read_bytes()
        if account in data: raise RuntimeError('Account-bearing source cannot be packaged: '+str(p.relative_to(ROOT)))
        rows.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    index=dict(schema=1,bundle_id='open-breakout-reliability-20261007-v2',
               created_utc=datetime.now(timezone.utc).isoformat(),files=rows,self_excluded=True)
    (ROOT/'package-integrity.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    output=args.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'package-integrity.json']: z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for r in rows: assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(digest+'  '+output.name+'\n',encoding='utf-8')
    print(json.dumps(dict(zip=str(output),sha256=digest,files=len(rows)+1),indent=2))


if __name__=='__main__': main()
