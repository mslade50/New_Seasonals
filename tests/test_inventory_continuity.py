import copy
import io
import json
import pandas as pd
import pytest
import cache_io
from actual_inventory_io import load_reviewed_seed
from scripts.harvest_fills import extend_canonical_coverage


def test_scanners_share_reviewed_r2_seed(monkeypatch, tmp_path):
    monkeypatch.delenv('TAGGED_INVENTORY_SEED', raising=False)
    monkeypatch.delenv('TAGGED_INVENTORY_SEED_R2_KEY', raising=False)
    seen = []
    class Client:
        def get_object(self, **kw):
            seen.append(kw)
            return {'Body': io.BytesIO(b'{"review":{"status":"approved"}}')}
    monkeypatch.setattr(cache_io, '_client', lambda: Client())
    monkeypatch.setattr(cache_io, '_r2_creds', lambda: {'R2_BUCKET': 'test'})
    assert load_reviewed_seed()['review']['status'] == 'approved'
    assert seen == [{'Bucket': 'test', 'Key': 'ops/tagged_inventory_seed.json'}]
    with pytest.raises(FileNotFoundError):
        load_reviewed_seed(tmp_path / 'missing.json')
    assert len(seen) == 1  # Explicit missing local input never changes sources.


@pytest.mark.parametrize('fault', [None, 'gap', 'account', 'unverified', 'incomplete'])
def test_canonical_history_remains_usable_after_live_retention_rolls(fault):
    old = dict(complete=True, broker_account='TEST', continuous_from='2026-08-01T00:00:00Z',
               complete_through='2026-09-09T20:00:00Z')
    new = dict(old, continuous_from='2026-08-10T00:00:00Z', complete_through='2026-09-10T20:00:00Z')
    prior = {'complete': True, 'completeness': {'accounts': {'primary': old}}}
    if fault == 'gap': new['continuous_from'] = '2026-09-09T21:00:00Z'
    if fault == 'account': old['broker_account'] = 'OTHER'
    if fault == 'incomplete': prior['complete'] = False
    frame = pd.DataFrame()
    if fault != 'unverified': frame.attrs['canonical_status'] = prior
    current = {'accounts': {'primary': new}}
    original = copy.deepcopy(current)
    result = extend_canonical_coverage(current, frame)
    assert result['accounts']['primary']['continuous_from'] == (old['continuous_from'] if fault is None else new['continuous_from'])
    assert current == original
