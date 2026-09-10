import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts import build_site


@pytest.fixture
def status(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "trading_day_offsets", lambda: (
        pd.offsets.CustomBusinessDay(), pd.Timestamp('2026-09-09'), pd.Timestamp('2026-09-08')))
    monkeypatch.setattr(build_site, "_ledger_provenance", lambda: {"build_utc": None})
    for name in ['MASTER_PRICES', 'EARNINGS', 'FRAGILITY', 'EXPOSURE_STATE']:
        monkeypatch.setattr(build_site, name, str(tmp_path / 'absent'))
    def read(frame=None):
        if frame is not None:
            frame.to_parquet(tmp_path / 'cboe_putcall.parquet')
        return build_site.build_health(None, str(tmp_path), build_id='test', built_at='2026-09-10T10:00:00Z')['artifacts']['cboe_putcall']
    return read


def ratios(dates):
    return pd.DataFrame({'equity': .55, 'total': .8, 'index': 1.1}, index=pd.to_datetime(dates))


def test_putcall_status_shows_complete_session_and_ratios(status):
    got = status(ratios(['2026-09-08', '2026-09-09']))
    assert got['status'] == 'fresh' and got['last_date'].startswith('2026-09-09')
    assert got['age_td'] == 0 and got['equity'] == .55 and got['index'] == 1.1


def test_old_putcall_does_not_become_fresh_when_copied(status):
    assert status(ratios(['2026-09-04']))['status'] == 'stale'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_incomplete_latest_putcall_row_is_not_green(status, value):
    frame = ratios(['2026-09-08', '2026-09-09']); frame.iloc[-1, 0] = value
    got = status(frame)
    assert got['status'] == 'stale' and got['last_date'].startswith('2026-09-08')
    assert 'incomplete' in got['note']


def test_future_and_duplicate_putcall_observations(status):
    assert status(ratios(['2026-09-08', '2026-09-10']))['status'] == 'stale'
    assert status(ratios(['2026-09-09', '2026-09-09']))['status'] == 'missing'


def test_missing_putcall_is_explicit(status):
    assert status()['status'] == 'missing'
