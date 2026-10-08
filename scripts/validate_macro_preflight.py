"""Read canonical R2 history; run the existing no-upload validator in isolation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KEY = 'macro_release_history.parquet'


def validate(client, bucket, *, root=ROOT, run_command=subprocess.run):
    # The client is used exclusively for HEAD/GET; no publisher/cache helper imports.
    stage = root / 'artifacts/macro_provider/preflight'
    stage.mkdir(parents=True, exist_ok=False)
    baseline = stage / 'prior.parquet'
    evidence = dict(validation_only=True, published=False, baseline_download_consistent=False,
                    canonical_baseline_unchanged=False, history_preservation_pass=False)
    receipt = stage / 'receipt.json'
    try:
        etag = client.head_object(Bucket=bucket, Key=KEY).get('ETag')
        if not etag:
            raise ValueError('canonical baseline has no ETag')
        response = client.get_object(Bucket=bucket, Key=KEY, IfMatch=etag)
        if response.get('ETag') != etag:
            raise ValueError('downloaded baseline identity mismatch')
        baseline.write_bytes(response['Body'].read())
        if client.head_object(Bucket=bucket, Key=KEY).get('ETag') != etag:
            raise ValueError('baseline changed during download')
        evidence.update(baseline_download_consistent=True,
                        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('R2_', 'AWS_'))}
        env['PYTHON_DOTENV_DISABLED'] = '1'
        output = stage / 'validation'
        command = [sys.executable, str(root / 'scripts/refresh_macro_releases.py'),
                   '--no-upload', '--baseline', str(baseline), '--output-dir', str(output)]
        print('mode=validation_only; collector=--no-upload; canonical/cache writes disabled', flush=True)
        result = run_command(command, cwd=root, env=env, check=False)
        evidence['canonical_baseline_unchanged'] = client.head_object(Bucket=bucket, Key=KEY).get('ETag') == etag
        if result.returncode or not evidence['canonical_baseline_unchanged']:
            raise ValueError('validation failed or canonical baseline changed')
        validated = json.loads((output / 'receipt.json').read_text())
        if validated.get('published') is not False or validated.get('official_series') != 29:
            raise ValueError('invalid no-upload validation receipt')
        evidence.update(history_preservation_pass=True, official_series=29,
                        rows=validated['rows'], preserved_populated_rows=validated['preserved_populated_rows'],
                        candidate_sha256=validated['candidate_sha256'])
        return 0
    except Exception as exc:
        evidence['error_type'] = type(exc).__name__
        return 1
    finally:
        receipt.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(evidence, sort_keys=True), flush=True)


def main():
    import boto3
    client = boto3.client('s3',
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'])
    return validate(client, os.environ['R2_BUCKET'])


if __name__ == '__main__':
    raise SystemExit(main())
