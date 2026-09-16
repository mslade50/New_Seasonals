import datetime as dt

from scripts import repo_health_check as health


def test_producer_status_is_expected_but_partial_write_still_warns(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'master_prices.parquet.status.json').write_text('{}')
    monkeypatch.setattr(health, 'ROOT', tmp_path)
    monkeypatch.setattr(health, '_last_index_date', lambda p: dt.date.today())
    reports = []
    monkeypatch.setattr(health, 'report', lambda *args: reports.append(args))
    health.check_local_data()
    assert reports[-1] == ('OK', 'data:stray-temp-files', 'none')
    (data / 'master_prices.parquet.tmp123').write_text('incomplete')
    health.check_local_data()
    assert reports[-1][0] == 'WARN'
    assert 'tmp123' in reports[-1][2]
    assert 'status.json' not in reports[-1][2]
