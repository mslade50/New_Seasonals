import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validate_macro_preflight import validate


@pytest.mark.parametrize('case', ['success', 'missing_etag', 'download_changed',
                                'head_changed', 'collector_failed', 'final_changed',
                                'published', 'coverage', 'unknown_read'])
def test_read_only_preflight_fails_closed_and_strips_credentials(tmp_path, monkeypatch, case):
    monkeypatch.setenv('R2_SECRET_ACCESS_KEY', 'never-in-child')
    monkeypatch.setenv('AWS_SESSION_TOKEN', 'never-in-child')
    calls = []

    class Reader:
        def head_object(self, **kwargs):
            calls.append('head')
            if case == 'unknown_read':
                raise RuntimeError('sensitive response must not be logged')
            n = calls.count('head')
            return {'ETag': None if case == 'missing_etag' else
                    'changed' if (case == 'head_changed' and n == 2) or
                    (case == 'final_changed' and n == 3) else 'original'}

        def get_object(self, **kwargs):
            calls.append('get')
            assert kwargs['IfMatch'] == 'original'
            return {'ETag': 'changed' if case == 'download_changed' else 'original',
                    'Body': io.BytesIO(b'canonical baseline')}

        def __getattr__(self, name):
            pytest.fail('preflight attempted non-read operation: ' + name)

    def collector(command, *, cwd, env, check):
        calls.append('collector')
        assert '--no-upload' in command and '--baseline' in command
        assert not any(k.startswith(('R2_', 'AWS_')) for k in env)
        assert env['PYTHON_DOTENV_DISABLED'] == '1'
        baseline = Path(command[command.index('--baseline') + 1])
        assert baseline.read_bytes() == b'canonical baseline'
        output = Path(command[command.index('--output-dir') + 1])
        output.mkdir()
        (output / 'receipt.json').write_text(json.dumps(dict(
            published=case == 'published', official_series=28 if case == 'coverage' else 29,
            rows=40, preserved_populated_rows=11, candidate_sha256='a' * 64)))
        return SimpleNamespace(returncode=1 if case == 'collector_failed' else 0)

    assert validate(Reader(), 'bucket', root=tmp_path, run_command=collector) == (0 if case == 'success' else 1)
    receipt = json.loads((tmp_path / 'artifacts/macro_provider/preflight/receipt.json').read_text())
    assert receipt['published'] is False and receipt['validation_only'] is True
    assert receipt['history_preservation_pass'] is (case == 'success')
    assert 'original' not in json.dumps(receipt) and 'sensitive' not in json.dumps(receipt)
    assert not (tmp_path / 'data').exists()


def test_workflow_default_publishes_only_explicit_validation_uses_preflight():
    workflow = (Path(__file__).parents[1] / '.github/workflows/build_macro_releases.yml').read_text()
    option = workflow.split('      validation_only:', 1)[1].split('      automation_token:', 1)[0]
    assert '        type: boolean' in option and '        default: false' in option
    steps = workflow.split('      - name: ')[1:]
    publish = next(s for s in steps if 'run: python scripts/refresh_macro_releases.py' in s)
    validation = next(s for s in steps if 'run: python scripts/validate_macro_preflight.py' in s)
    assert '        if: ${{ !inputs.validation_only }}' in publish
    assert '        if: ${{ inputs.validation_only }}' in validation
    assert publish.split('        env:', 1)[1].split('        run:', 1)[0] == validation.split('        env:', 1)[1].split('        run:', 1)[0]


def test_real_no_upload_runner_preserves_cache_and_history(tmp_path, monkeypatch):
    import pandas as pd
    from official_macro_releases import observation
    from scripts import refresh_macro_releases as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    old = observation('initial_jobless_claims', 210, '2026-09-12', '2026-09-17T12:30:00Z',
                      source='https://www.dol.gov/ui/data.pdf', fetched_at='2026-09-24T15:00:00Z',
                      digest='abc', unit='K')
    baseline = tmp_path / 'baseline.parquet'
    pd.DataFrame([old]).to_parquet(baseline)
    original = baseline.read_bytes()
    cache = tmp_path / 'data/macro_release_history.parquet'
    cache.parent.mkdir()
    cache.write_bytes(b'untouched local cache')
    def collect(output, **kwargs):
        pd.DataFrame([old]).to_parquet(output / 'official_latest.parquet')
        (output / 'manifest.json').write_text('{}')
        return dict(core_data_pass=True, publication_eligible=True, unique_series=29, warnings=[])
    monkeypatch.setattr(runner, 'collect', collect)
    monkeypatch.setattr(runner, 'publish', lambda *a: pytest.fail('no-upload invoked publisher'))
    out = tmp_path / 'artifacts/validated'
    assert runner.main(['--no-upload', '--baseline', str(baseline), '--output-dir', str(out)]) == 0
    assert baseline.read_bytes() == original and cache.read_bytes() == b'untouched local cache'
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['published'] is False and receipt['preserved_populated_rows'] == 1
