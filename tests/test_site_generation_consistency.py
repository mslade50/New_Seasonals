import json
from pathlib import Path

import pytest

from scripts import site_r2_pipeline as pipeline


def prepare(tmp_path, monkeypatch):
    objects = {'prices.json': b'{"price":100}'}
    canonical = pipeline.R2Input('prices', 'prices.json', 'data/prices.json')
    generated = pipeline.R2Input('ledger', 'ledger.json', 'data/ledger.json')
    monkeypatch.setattr(pipeline, 'CANONICAL_INPUTS', (canonical,))
    monkeypatch.setattr(pipeline, 'GENERATED_INPUTS', (generated,))
    monkeypatch.setattr(pipeline, '_require_cloud_stage', lambda root, **kw: {'source_sha': root.name.split('-')[0]})

    def download(key, destination):
        if key not in objects:
            return False
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(objects[key])
        return True

    def upload(source, key):
        objects[key] = Path(source).read_bytes()
        return True

    monkeypatch.setattr(pipeline.cache_io, 'download_to_local', download)
    monkeypatch.setattr(pipeline.cache_io, 'upload_from_local', upload)
    monkeypatch.setattr(pipeline.cache_io, 'head', lambda key: {'ContentLength': len(objects[key]), 'ETag': 'fixture'})
    generator = tmp_path / 'same-generator'
    pipeline.pull_generator(generator)
    (generator / 'data/ledger.json').write_text('{"computed_from":100}')
    return objects, generator


def test_assembler_uses_original_canonical_generation(tmp_path, monkeypatch):
    objects, generator = prepare(tmp_path, monkeypatch)
    bundle = pipeline.publish_generated(generator, 'fixture')
    objects['prices.json'] = b'{"price":200}'
    assembler = tmp_path / 'same-assembler'
    pipeline.pull_assembler(assembler, 'fixture')
    assert json.loads((assembler / 'data/prices.json').read_text())['price'] == 100
    assert bundle['inputs'][0]['sha256']


def test_assembler_rejects_different_source_revision(tmp_path, monkeypatch):
    _, generator = prepare(tmp_path, monkeypatch)
    pipeline.publish_generated(generator, 'fixture')
    with pytest.raises(RuntimeError, match='source'):
        pipeline.pull_assembler(tmp_path / 'different-assembler', 'fixture')


def test_publication_rejects_mutated_generator_input(tmp_path, monkeypatch):
    _, generator = prepare(tmp_path, monkeypatch)
    (generator / 'data/prices.json').write_text('{"price":200}')
    with pytest.raises(RuntimeError, match='input.*changed'):
        pipeline.publish_generated(generator, 'fixture')
