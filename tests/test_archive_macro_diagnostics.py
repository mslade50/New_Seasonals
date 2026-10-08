import json

import pytest

import scripts.archive_macro_diagnostics as diagnostics


def test_sanitizer_keeps_gate_and_clock_but_excludes_sensitive_fields():
    value = dict(captured_at='2026-10-08T01:57:42+00:00', core_data_pass=False,
                 published=False, fmp_requests=0, unique_series=29,
                 gaps=['missed next announced release: pce',
                       'BEA pce: HTTPError: secret https://host/?token=secret',
                       'token=secret'], warnings=['bls_feed.html: HTTPError: secret'],
                 sources=[{'url': 'https://host/?token=secret'}],
                 error='secret', baseline_etag='secret', provider_observation='secret')
    clean = diagnostics.sanitize(value)
    assert clean['unique_series'] == 29 and clean['core_data_pass'] is False
    assert 'missed next announced release: pce' in clean['gaps']
    assert 'BEA pce: HTTPError (details omitted)' in clean['gaps']
    assert 'secret' not in json.dumps(clean) and 'https://' not in json.dumps(clean)
    assert diagnostics.reason('stale release: government_payrolls') == 'stale release: government_payrolls'
    assert diagnostics.reason('BEA pce: SecretError: secret') == 'unclassified gate failure'


def test_wrong_types_and_unvalidated_text_are_not_retained():
    clean = diagnostics.sanitize(dict(published='secret', rows=True, captured_at='secret',
                                     candidate_sha256='secret', error_type='SecretError', warnings=['BEA pce: token=secret']))
    assert clean == {'warnings': ['unclassified gate failure']}


def test_archive_does_not_copy_raw_files_or_parquets(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'ROOT', tmp_path)
    source = tmp_path / 'artifacts/macro_provider/run/official'
    source.mkdir(parents=True)
    (source / 'manifest.json').write_text(json.dumps({'gaps': ['missed next announced release: pce'], 'raw': 'secret'}))
    (source / 'source.html').write_text('secret')
    (source / 'prior.parquet').write_bytes(b'secret')
    output = tmp_path / 'artifacts/report'
    result = diagnostics.archive(source.parent.parent, output)
    assert list(output.iterdir()) == [result]
    records = json.loads(result.read_text())['records']
    assert len(records) == 1 and records[0]['metadata']['gaps'] == ['missed next announced release: pce']
    assert 'secret' not in result.read_text()


def test_absent_inputs_and_invalid_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'ROOT', tmp_path)
    result = diagnostics.archive(tmp_path / 'artifacts/absent', tmp_path / 'artifacts/report')
    assert json.loads(result.read_text())['records'] == []
    with pytest.raises(ValueError):
        diagnostics.archive(tmp_path, tmp_path / 'artifacts/unsafe')
    with pytest.raises(ValueError):
        diagnostics.archive(tmp_path / 'artifacts/absent', tmp_path / 'artifacts/absent/out')


@pytest.mark.parametrize('state', [
    'attempting_conditional_write', 'baseline_conflict', 'write_outcome_unknown',
    'remote_written_unverified', 'remote_verified',
])
def test_publication_state_allowlist_is_preserved_independently_of_published(state):
    clean = diagnostics.sanitize({'published': False, 'publication_state': state,
                                  'published_etag': 'secret', 'error': 'secret'})
    assert clean == {'published': False, 'publication_state': state}


@pytest.mark.parametrize('state', ['secret', None, {'token': 'secret'}, ['remote_verified']])
def test_unrecognized_or_wrong_type_publication_state_is_omitted(state):
    assert diagnostics.sanitize({'published': False, 'publication_state': state}) == {'published': False}


def test_no_publication_attempt_does_not_gain_a_remote_state():
    clean = diagnostics.sanitize({'published': False, 'error_type': 'ValueError',
                                 'error': 'official coverage gate failed; see archived manifest'})
    assert clean['published'] is False and 'publication_state' not in clean
    attempted = diagnostics.sanitize({'published': False, 'publication_state': 'remote_written_unverified'})
    assert attempted != clean and attempted['publication_state'] == 'remote_written_unverified'


@pytest.mark.parametrize('outcome,expected', [
    ('conflict', 'baseline_conflict'),
    ('unknown', 'write_outcome_unknown'),
    ('readback_failed', 'remote_written_unverified'),
    ('verified', 'remote_verified'),
])
def test_mocked_publisher_state_survives_archive(tmp_path, monkeypatch, outcome, expected):
    import cache_io
    from scripts import refresh_macro_releases as publisher

    monkeypatch.setattr(diagnostics, 'ROOT', tmp_path)
    run = tmp_path / 'artifacts/macro_provider/run'
    run.mkdir(parents=True)
    candidate = run / 'candidate.parquet'
    candidate.write_bytes(b'candidate')
    local = run / 'local.parquet'
    local.write_bytes(b'prior')
    writes, reads = [], []

    def upload(path, key, **kwargs):
        writes.append((path, key))
        assert kwargs['expected_etag'] == 'test-baseline'
        state = 'precondition_failed' if outcome == 'conflict' else 'unknown' if outcome == 'unknown' else 'uploaded'
        return state, 'secret-etag'

    def download(key, path):
        from pathlib import Path
        reads.append(key)
        Path(path).write_bytes(b'bad' if outcome == 'readback_failed' else b'candidate')
        return True

    monkeypatch.setattr(cache_io, 'conditional_upload_from_local', upload)
    monkeypatch.setattr(cache_io, 'download_to_local', download)
    receipt = {'published': False, 'baseline_etag': 'secret-baseline'}
    if outcome == 'verified':
        publisher.publish(candidate, 'test-baseline', local, run, receipt)
        assert local.read_bytes() == b'candidate'
    else:
        with pytest.raises(ValueError):
            publisher.publish(candidate, 'test-baseline', local, run, receipt)
        assert local.read_bytes() == b'prior'
    assert len(writes) == 1 and len(reads) == int(outcome in {'readback_failed', 'verified'})
    result = diagnostics.archive(run.parent, tmp_path / 'artifacts/report')
    records = json.loads(result.read_text())['records']
    assert len(records) == 1 and records[0]['kind'] == 'publication.json'
    assert records[0]['metadata'] == {'published': False, 'publication_state': expected}
    assert 'secret' not in result.read_text()
