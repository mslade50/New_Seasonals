"""Prepare a hash-pinned read-only book upgrade. Never installs or connects."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from broker_runtime.prepare import replace_once, patch_olv

HASHES = {
    'book_snapshot.py':'6a01d281d38c35352fa310ca6908201b063f9a9d784ad6a903697fd33a95060a',
    'olv_exit_moo.py':'3e8c1ad947cec79addc6773669155ce625b35dcb6a1795773e479c699d74dca3',
}


def patch_book(source):
    source = replace_once(source, 'import json\n', 'import json\nimport time\n')
    source = replace_once(source, '"qty": _num(o.totalQuantity), "order_type": o.orderType,',
        '"qty": _num(o.totalQuantity), "order_type": o.orderType,\n'
        '                "remaining": _num(t.orderStatus.remaining), "filled": _num(t.orderStatus.filled),')
    source = replace_once(source, "        # Today's executions.",
        '        out["orders_source_at"] = time.time()\n\n        # Today\'s executions.')
    ast.parse(source)
    return source


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('candidate destination must be new')
    bodies = {}
    for name, expected in HASHES.items():
        body = (source/name).read_bytes()
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError(f'reviewed source changed: {name}')
        bodies[name] = body.decode('utf-8-sig')
    candidates = {'book_snapshot.py':patch_book(bodies['book_snapshot.py']),
                  'olv_exit_moo.py':patch_olv(bodies['olv_exit_moo.py']),
                  'olv_contract.py':Path(__file__).with_name('olv_contract.py').read_text(encoding='utf-8')}
    for name, body in candidates.items():
        compile(body, name, 'exec')
    output.mkdir(parents=True)
    manifest = {'source_sha256':HASHES,'candidate_sha256':{},
                'activation':'not installed; OLV exit candidate is Primary-only and requires coordinated PA/inventory cutover'}
    for name, body in candidates.items():
        data = body.encode('utf-8')
        (output/name).write_bytes(data)
        manifest['candidate_sha256'][name] = hashlib.sha256(data).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.output),indent=2))
