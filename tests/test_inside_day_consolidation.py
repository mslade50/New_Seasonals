import numpy as np
import pandas as pd

from indicators import calculate_indicators, consolidation_features
from scripts.backtest_inside_day_consolidation import annual_count_gate, episode_mask, profile_mask


def test_cap_is_per_year_not_average_or_number_of_fills():
    years = np.r_[np.zeros(101, int), np.ones(150, int)]
    ok, counts = annual_count_gate(years, np.ones(len(years), bool))
    assert not ok and counts[0] == 101 and counts[1] == 150
    selected = np.ones(len(years), bool)
    selected[0] = False
    selected[101:151] = False
    ok, counts = annual_count_gate(years, selected, min_signals=200)
    assert ok and counts.max() == 100


def test_episode_spacing_uses_prior_raw_matches_not_last_accepted():
    # Qualifying X signals at 1/5/12 form one 10-session cluster, followed by
    # a new episode at 22. Signals at 12 and 22 are ten bars apart exactly.
    ticker = np.array([0, 0, 0, 0, 1, 1])
    bars = np.array([1, 5, 12, 22, 1, 11])
    assert episode_mask(np.ones(6, bool), ticker, bars, 10).tolist() == [True, False, False, True, True, True]


def test_unfilled_signals_count_toward_the_limit():
    # No event_idx is used by the count gate: all 101 signals count even
    # if just 10 happen to execute tomorrow.
    years = np.zeros(101, int)
    selected = np.ones(101, bool)
    assert not annual_count_gate(years, selected, min_signals=0)[0]


def test_features_are_causal_and_zero_volume_is_unavailable():
    rng = np.random.default_rng(2)
    c = 100 * np.exp(np.cumsum(rng.normal(.001, .01, 620)))
    raw = pd.DataFrame({'Open': c, 'High': c*1.01, 'Low': c*.99, 'Close': c,
                        'Volume': np.full(620, 1e6)}, index=pd.bdate_range('2000-01-03', periods=620))
    raw.loc[raw.index[580], 'Volume'] = 0
    short = consolidation_features(calculate_indicators(raw.iloc[:600], {}, 'X'))
    full = consolidation_features(calculate_indicators(raw, {}, 'X'))
    pd.testing.assert_frame_equal(short, full.iloc[:600])
    assert np.isnan(full.quiet_volume_ratio.iloc[580])
    assert full.quiet_volume_ratio.iloc[579] == 1.


def test_volume_filter_does_not_allow_missing_zero_volume():
    features = pd.DataFrame({'range5_atr': [1., 1.], 'distance_high252_pct': [2., 2.],
                             'trend_stack': [True, True], 'quiet_volume_ratio': [.5, np.nan],
                             'day_range_atr': [.4, .4]})
    profile = {'family': 'quiet_inside', 'range5': 1.5, 'near_high_pct': 3.,
               'trend': 'trend_stack', 'volume': .6, 'use126': False,
               'exclude_new_close_high': False, 'special': .5}
    assert profile_mask(features, profile).tolist() == [True, False]
