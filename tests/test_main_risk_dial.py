"""One stored score selects the sample for every forward-return window."""
import pandas as pd

from fragility_core import load_main_dial_series, load_pit_sizing_state
from daily_risk_report import build_main_dial_forward_returns, _build_fwd_returns_html
from pages.risk_dashboard_v2 import compute_similar_reading_returns
from scripts.build_risk_json import build_nuggets


def test_main_returns_match_displayed_score_and_keep_all_windows(tmp_path):
    dates = pd.bdate_range('2024-01-02', periods=320)
    # Two deliberately conflicting unused models catch accidental selection.
    stored = pd.DataFrame({'5d': 99., '21d': 1., '63d': 55.}, index=dates)
    stored.loc[dates[-10:], '63d'] = list(range(46, 66, 2))
    path = tmp_path / 'dial.parquet'
    stored.to_parquet(path)
    prices = pd.Series(100 + pd.RangeIndex(len(dates)) * .2, index=dates)
    series = load_main_dial_series(path)
    state = load_pit_sizing_state(path=str(path), asof=dates[-1])
    assert series.iloc[-1] == state['score'] == 55.
    result = build_main_dial_forward_returns(prices, path)
    assert set(result) == {'63d'}
    expected = compute_similar_reading_returns(series, prices, state['score'])
    assert result['63d']['episode_dates'] == expected['episode_dates']
    assert result['63d']['current_score'] == state['score']
    assert set(result['63d']['returns']) == {5, 10, 21, 42, 63}
    assert all(stats is not None for stats in result['63d']['returns'].values())


def test_missing_main_history_does_not_substitute_other_dials(tmp_path):
    path = tmp_path / 'missing.parquet'
    assert load_main_dial_series(path) is None
    pd.DataFrame({'5d': [20.], '21d': [40.]}).to_parquet(path)
    assert build_main_dial_forward_returns(pd.Series(dtype=float), path) == {}


def test_email_and_nuggets_show_all_windows_from_one_score():
    stats = {'mean': .01, 'median': .01, 'pct_neg': .3, 'mean_z': .2,
             'median_z': .1, 'uncond_mean': .005}
    record = {'current_score': 51., 'n_episodes': 20, 'band_low': 46.,
              'band_high': 56., 'returns': {n: stats for n in [5, 10, 21, 42, 63]}}
    html = _build_fwd_returns_html({'63d': record, '5d': record, '21d': record}, 'Returns')
    assert html.count('Main risk dial =') == 1
    for window in [5, 10, 21, 42, 63]:
        assert f'>{window}d</td>' in html
    record['returns'] = {str(n): stats for n in [5, 10, 21, 42, 63]}
    notes = build_nuggets({'sizing_state': {'score': 51.}, 'forward_returns': {'63d': record}})
    lines = '\n'.join(line for note in notes for line in note['lines'])
    for window in [5, 21, 63]:
        assert f'SPY next {window}d averaged' in lines
    assert '21d score' not in lines
