"""Capture verified Primary OLV holdings, pending entries and NAV at 16:05 ET."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actual_inventory_io import load_actual_inventory
from closing_inventory import STRATEGY, make_snapshot, snapshot_key
import olv_capacity


def main(argv=None, *, now=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('data/olv_closing_inventory.json'))
    parser.add_argument('--capacity-output', type=Path, default=Path('data/olv_closing_capacity.json'))
    args = parser.parse_args(argv)
    # Capacity does not depend on a seed, execution history, or exit metadata.
    book = olv_capacity.query_local_book()
    capacity_snapshot = olv_capacity.make_snapshot(book, now=now or pd.Timestamp.now(tz='UTC'))
    capacity_body = (json.dumps(capacity_snapshot, sort_keys=True, allow_nan=False)+'\n').encode()
    if args.publish:
        from cache_io import _client, _r2_creds
        client, creds = _client(), _r2_creds()
        if client is None or creds is None:
            raise RuntimeError('closing capacity storage is unavailable')
        digest = hashlib.sha256(capacity_body).hexdigest()
        client.put_object(Bucket=creds['R2_BUCKET'], Key=f'ops/olv_capacity/generations/{digest}.json', Body=capacity_body)
        client.put_object(Bucket=creds['R2_BUCKET'], Key=olv_capacity.snapshot_key(capacity_snapshot['session']), Body=capacity_body)
        client.put_object(Bucket=creds['R2_BUCKET'], Key='ops/olv_capacity/latest.json', Body=capacity_body)
    args.capacity_output.parent.mkdir(parents=True, exist_ok=True)
    args.capacity_output.write_bytes(capacity_body)
    inventory = load_actual_inventory(algo_strategies={STRATEGY})
    if inventory.status != 'known':
        print(json.dumps(dict(session=capacity_snapshot['session'], capacity='known',
                              inventory='unknown', reasons=inventory.reasons, published=args.publish)))
        return
    snapshot = make_snapshot(inventory, now=now or pd.Timestamp.now(tz='UTC'))
    body = (json.dumps(snapshot, sort_keys=True, allow_nan=False) + '\n').encode()
    if args.publish:
        from cache_io import _client, _r2_creds
        client, creds = _client(), _r2_creds()
        if client is None or creds is None:
            raise RuntimeError('closing inventory storage is unavailable')
        # Keep the full captured evidence before publishing the dated reader key.
        digest = hashlib.sha256(body).hexdigest()
        client.put_object(Bucket=creds['R2_BUCKET'], Key=f'ops/olv_closing_inventory/generations/{digest}.json', Body=body)
        client.put_object(Bucket=creds['R2_BUCKET'], Key=snapshot_key(snapshot['session']), Body=body)
        client.put_object(Bucket=creds['R2_BUCKET'], Key='ops/olv_closing_inventory/latest.json', Body=body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(body)
    print(json.dumps(dict(session=snapshot['session'], observed_at=snapshot['observed_at'],
                          tranches=len(inventory.tranches), published=args.publish)))


if __name__ == '__main__':
    main()
