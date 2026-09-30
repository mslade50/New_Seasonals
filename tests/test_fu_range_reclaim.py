from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scratch.fu_range_reclaim.research import Rules, detect, features
from scratch.fu_range_reclaim.execution import simulate, evaluate


def pattern():
    before = np.linspace(70, 106, 240)
    range_close = 113-2*np.cos(2*np.pi*(np.arange(40)-3)/12)
    close = np.r_[before, range_close, 112.5, np.full(25, 114.)]
    d = pd.DataFrame({'Open': close-.2, 'High': close+1, 'Low': close-1, 'Close': close},
                     index=pd.bdate_range('2020-01-01', periods=len(close)))
    d.iloc[280] = [111.5, 113., 109.6, 112.5]
    return d


def test_established_range_above_rising_200():
    d = pattern()
    s = detect(d)
    assert len(s) == 1
    q = s[0]
    assert q['signal_i'] == 280
    assert q['support_tests'] == [243, 255, 267]
    assert q['resistance_tests'] == [249, 261]
    assert q['support'] == pytest.approx(110.)
    assert q['resistance'] == pytest.approx(116.)
    assert q['known_i'] == 274
    assert q['support'] > q['sma200_prior'] > q['sma200_20ago']


def test_prefix_invariance_and_future_mutation():
    d = pattern()
    all_signals = detect(d)
    for end in range(250, len(d)+1):
        assert detect(d.iloc[:end]) == [s for s in all_signals if s['signal_i'] < end]
    d.iloc[281:] *= .2
    assert [s for s in detect(d) if s['signal_i'] <= 280] == all_signals


def test_requires_rising_200_even_when_price_is_above_it():
    d = pattern()
    d.iloc[:102] = np.array([139.8, 141., 139., 140.])
    f = features(d)
    assert d.Close.iloc[280] > f.sma200.iloc[280]
    assert f.sma200.iloc[279] < f.sma200.iloc[259]
    assert detect(d) == []


def test_requires_price_and_floor_above_200():
    d = pattern()
    d.iloc[:240] += 45
    assert features(d).sma200.iloc[280] > d.Close.iloc[280]
    assert detect(d) == []


def test_rising_20day_comparison_cannot_mask_a_current_downturn():
    d = pattern()
    d.iloc[80] = [149.8, 151., 149., 150.]
    f = features(d)
    assert f.sma200.iloc[279] > f.sma200.iloc[259]
    assert f.sma200.iloc[280] < f.sma200.iloc[279]
    assert d.Close.iloc[280] > f.sma200.iloc[280]
    assert detect(d) == []


def test_one_low_is_not_a_defined_range():
    d = pattern()
    for i in [243, 255]:
        d.loc[d.index[i], 'Low'] = 111.7
    assert detect(d) == []


def test_line_cannot_be_redefined_just_before_signal():
    d = pattern()
    d.loc[d.index[278], 'Low'] = 109.9
    assert detect(d) == []


def test_pivots_must_be_confirmed_before_frozen_period():
    d = pattern()
    # There are three total ceiling pivots, but the last is not confirmed
    # at the cutoff. Requiring three must therefore reject this setup.
    assert detect(d, replace(Rules(), resistance_tests=3)) == []


@pytest.mark.parametrize('low,close', [(108., 112.5), (109.6, 109.9), (110., 112.5)])
def test_deep_sweep_failed_reclaim_or_no_undercut_rejected(low, close):
    d = pattern()
    d.loc[d.index[280], ['Low', 'Close']] = [low, close]
    assert detect(d) == []


def test_scale_invariance_and_next_open_horizon():
    d = pattern()
    s, s2 = detect(d)[0], detect(d*.6)[0]
    assert s['signal_i'] == s2['signal_i']
    t, t2 = simulate(d, s), simulate(d*.6, s2)
    assert t['entry_i'] == 281 and t['exit_i'] == 290
    assert t['net'] == pytest.approx(t2['net'])
    assert simulate(d.iloc[:290], s) is None


def test_stop_entry_day_gap_and_invalid_entry():
    d = pattern()
    s = detect(d)[0]
    level = s['sweep_low']-.1*s['atr_ref']
    d.loc[d.index[281], 'Low'] = level-.1
    assert simulate(d, s, stop=True)['reason'] == 'stop'
    d.loc[d.index[281], 'Low'] = 113
    d.loc[d.index[282], ['Open', 'Low']] = [level-2, level-3]
    q = simulate(d, s, stop=True)
    assert q['reason'] == 'gap_stop' and q['exit'] == pytest.approx(level-2)
    d.loc[d.index[281], 'Open'] = level
    assert simulate(d, s, stop=True) is None


def test_execution_nonoverlap():
    d = pattern()
    s = detect(d)[0]
    rows = evaluate(d, [s, {**s, 'signal_i': 282}, {**s, 'signal_i': 290}])
    assert [q['signal_i'] for q in rows] == [280, 290]
