"""Prepare only the read-only snapshot upgrade; no order runner or live writes."""
import argparse
import hashlib
import json
from pathlib import Path
from broker_runtime.prepare import patch_snapshot
from broker_runtime.prepare_olv_capacity import HASHES, patch_book


def prepare(source, output):
    source, output = Path(source), Path(output)
    body=(source/'book_snapshot.py').read_bytes()
    expected=HASHES['book_snapshot.py']
    if hashlib.sha256(body).hexdigest()!=expected:
        raise ValueError('reviewed book_snapshot.py source changed')
    candidate=patch_snapshot(patch_book(body.decode('utf-8-sig'))).encode('utf-8')
    compile(candidate,'book_snapshot.py','exec')
    output.mkdir(parents=True,exist_ok=False)
    (output/'book_snapshot.py').write_bytes(candidate)
    manifest={'source_sha256':expected,'candidate_sha256':hashlib.sha256(candidate).hexdigest(),
              'activation':'not installed; read-only Primary and PA snapshot only; no trading runner changes',
              'coverage':'completed current-session query only; no historical continuity inferred'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.output),indent=2))
