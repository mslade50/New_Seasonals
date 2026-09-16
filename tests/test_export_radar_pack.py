import pandas as pd
import pytest

from scripts import export_radar_pack as exporter
from trading_calendar import TRADING_DAY


def test_cross_year_export_preserves_source_and_reports_incomplete_tickers(tmp_path, monkeypatch):
    asof = pd.Timestamp('2026-09-12')
    sessions = pd.date_range(asof + pd.Timedelta(days=1), periods=90, freq=TRADING_DAY)
    current = sessions[sessions.year == 2026]
    source = pd.concat([pd.DataFrame({
        'ticker': ticker, 'Date': current, 'atr_sznl_21d': 60., 'atr_sznl_63d': 70.,
    }) for ticker in ('AAA', 'MISSING')])
    path = tmp_path / 'ranks.parquet'
    source.to_parquet(path)
    before = path.read_bytes()
    monkeypatch.setattr(exporter, 'ATR_RANKS_PARQUET', path)
    monkeypatch.setattr(exporter, 'build_future_year_projection', lambda u, a, dates: pd.DataFrame({
        'ticker': 'AAA', 'Date': dates, 'atr_sznl_21d': 55., 'atr_sznl_63d': 65.,
    }))
    out, missing, actual_sessions = exporter.build_atr_sznl_csv(['AAA', 'MISSING'], asof)
    assert len(out) == 90
    assert missing == ['MISSING']
    assert actual_sessions.equals(sessions)
    assert out.attrs['projected_years'] == [2027]
    assert path.read_bytes() == before


def test_projection_cuts_prices_before_computing_forward_outcomes(monkeypatch):
    import build_atr_seasonal_ranks as ranks

    asof = pd.Timestamp('2026-09-12')
    frame = pd.DataFrame({'Close': [10., 999.]}, index=pd.to_datetime(['2026-09-11', '2026-09-14']))
    monkeypatch.setattr(ranks, '_read_price_parquet', lambda p, wanted: {'AAA': frame} if wanted else {})
    observed = []
    monkeypatch.setattr(ranks, 'prepare_ticker_data', lambda f: observed.append(f.copy()) or f)
    monkeypatch.setattr(ranks, 'compute_ranks_for_year', lambda f, year: pd.DataFrame({
        'atr_sznl_21d': [60.], 'atr_sznl_63d': [70.],
    }, index=[1]))
    monkeypatch.setattr(ranks, 'generate_trading_dates', lambda year: pd.DataFrame({
        'Date': [pd.Timestamp('2027-01-04')], 'day_count': [1],
    }))
    out = exporter.build_future_year_projection(['AAA'], asof, pd.DatetimeIndex(['2027-01-04']))
    assert len(observed[0]) == 1
    assert observed[0].index.max() <= asof
    assert out.iloc[0]['atr_sznl_63d'] == 70.


def test_missing_current_year_still_fails(tmp_path, monkeypatch):
    path = tmp_path / 'ranks.parquet'
    pd.DataFrame({'ticker': ['AAA'], 'Date': pd.to_datetime(['2025-12-31']),
                  'atr_sznl_21d': [50.], 'atr_sznl_63d': [50.]}).to_parquet(path)
    monkeypatch.setattr(exporter, 'ATR_RANKS_PARQUET', path)
    with pytest.raises(SystemExit, match='lack the current year'):
        exporter.build_atr_sznl_csv(['AAA'], pd.Timestamp('2026-09-12'))
