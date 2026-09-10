"""Read Primary only with a client ID separate from the site's collector.

The installed snapshot owns the broker query and account resolution. This
adapter does not start the command agent or invoke any trading runner.
"""
import importlib.util
import json
from pathlib import Path
import sys


def query(snapshot_path):
    path = Path(snapshot_path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('inventory_readonly_snapshot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    primary = [a for a in module.ACCOUNTS if a.get('key') == 'primary']
    if len(primary) != 1:
        raise ValueError('snapshot must configure exactly one Primary endpoint')
    return {'accounts': [module.snap_account(dict(primary[0], cid=8122))]}


if __name__ == '__main__':
    print(json.dumps(query(sys.argv[1])))
