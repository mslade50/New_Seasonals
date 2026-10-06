import numpy as np
import pandas as pd

from indicators import consolidation_audit_features
from scripts.backtest_inside_day_filter_audit import definitions, selected_mask, daily_mtm, cache_covers_rules


def feature_input(n=300):
    index = pd.bdate_range('2025-01-01', periods=n)
    return pd.DataFrame({'Close': 100., 'High': 101., 'Low': 99., 'Volume': 100.,
                         'ATR': 2., 'SMA50': 98., 'SMA200': 95.}, index=index)


def test_half_session_is_not_automatically_low_volume():
    df = feature_input()
    hours = pd.Series(6.5, index=df.index)
    hours.iloc[-1] = 3.5
    df.loc[df.index[-1], 'Volume'] = 30.
    f = consolidation_audit_features(df, hours)
    assert f.quiet_volume_ratio.iloc[-1] < .4
    assert f.quiet_volume_rate_ratio.iloc[-1] > .4


def test_session_features_are_causal_and_missing_hours_fail_closed():
    df = feature_input()
    hours = pd.Series(6.5, index=df.index)
    hours.iloc[270] = np.nan
    short = consolidation_audit_features(df.iloc[:280], hours.iloc[:280])
    full = consolidation_audit_features(df, hours)
    pd.testing.assert_frame_equal(short, full.iloc[:280])
    assert np.isnan(full.quiet_volume_rate_ratio.iloc[270])


def test_original_gate_ablation_admits_previously_excluded_candidates():
    rules = {r['name']: r for r in definitions()}
    flags = pd.DataFrame(True, index=[0, 1], columns=rules['baseline']['required'])
    flags.loc[1, 'rank252'] = False
    assert selected_mask(flags, rules['baseline']['required']).tolist() == [True, False]
    assert selected_mask(flags, rules['base_drop_rank252']['required']).tolist() == [True, True]
    assert len(rules) == 167


def test_cached_union_cannot_admit_a_new_broader_rule():
    loose = [{'required': ['quiet']}]
    tight = [{'required': ['quiet', 'inside']}]
    assert cache_covers_rules(loose, tight, ['quiet', 'inside'])
    assert not cache_covers_rules(tight, loose, ['quiet', 'inside'])
    assert not cache_covers_rules(loose, tight, ['quiet'])


def test_daily_marks_charge_both_fees_and_reconcile():
    sessions = pd.bdate_range('2026-01-05', periods=3)
    meta = pd.DataFrame({'entry': [100.], 'atr': [2.]})
    paths = {'Close': np.array([[101., 102., 103.]]), 'date': np.array([sessions.to_numpy()])}
    net = (103*.9995-100*1.0005)/2
    result = pd.DataFrame({'net_atr': [net]})
    pnl, live = daily_mtm(meta, result, paths, 2, sessions)
    np.testing.assert_allclose(pnl, [.475, .5, .47425])
    assert live.all() and abs(pnl.sum()-net)<1e-10
