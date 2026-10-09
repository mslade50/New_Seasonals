import json

import numpy as np
import pandas as pd

from indicators import consolidation_audit_features
from scripts.backtest_inside_day_filter_audit import definitions, selected_mask, daily_mtm, cache_covers_rules
from scripts.backtest_inside_day_filter_audit import (
    consolidation_breakout_mask, selected_definition, SELECTED_HOLD_DAYS, write_selected_outputs,
    market_open_hours,
)


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


def test_selected_breakout_uses_per_hour_volume_and_fails_closed():
    df = feature_input()
    df['Close'] = 100.4
    df['SMA50'] = np.linspace(97., 98., len(df))
    df['SMA10'], df['EMA21'] = 99.8, 99.6
    for window in (252, 5, 10, 21):
        df[f'rank_ret_{window}d'] = 60.
    df.loc[df.index[-1], ['High', 'Low', 'Close', 'Volume']] = [100.7, 99.3, 100.2, 30.]
    hours = pd.Series(6.5, index=df.index)
    assert consolidation_breakout_mask(df, hours).iloc[-1]
    hours.iloc[-1] = 3.5
    assert not consolidation_breakout_mask(df, hours).iloc[-1]
    hours.iloc[-1] = np.nan
    assert not consolidation_breakout_mask(df, hours).iloc[-1]
    required = selected_definition()['required']
    assert 'quiet_rate' in required and 'quiet' not in required
    assert SELECTED_HOLD_DAYS == 10


def test_known_nyse_hours_include_early_close_and_exclude_holiday():
    hours = market_open_hours()
    assert hours.loc['2025-11-26'] == 6.5
    assert hours.loc['2025-11-28'] == 3.5
    assert pd.Timestamp('2025-11-27') not in hours.index


def test_selected_exports_choose_the_approved_signal_and_ten_day_exit(tmp_path):
    rows = pd.DataFrame([
        {'rule': 'baseline', 'hold': 10, 'signals': 2, 'trades': 2},
        {'rule': 'session_adjusted_volume', 'hold': 5, 'signals': 1, 'trades': 1},
        {'rule': 'session_adjusted_volume', 'hold': 10, 'signals': 1, 'trades': 1},
    ])
    signals = pd.DataFrame({'rule': ['baseline', 'baseline', 'session_adjusted_volume'],
                            'ticker': ['X', 'Y', 'X']})
    trades = pd.DataFrame({'rule': ['baseline', 'session_adjusted_volume', 'session_adjusted_volume'],
                           'hold': [10, 5, 10], 'ticker': ['Y', 'Y', 'X']})
    write_selected_outputs(rows, signals, trades, tmp_path)
    assert pd.read_csv(tmp_path / 'selected_signals.csv').ticker.tolist() == ['X']
    assert pd.read_parquet(tmp_path / 'selected_trades.parquet').ticker.tolist() == ['X']
    selected = json.loads((tmp_path / 'selected_strategy.json').read_text())
    assert selected['volume_basis'] == 'per_market_open_hour'
    assert selected['volume_ceiling'] == .4
    assert selected['hold_days'] == 10 and selected['stop'] is None and selected['target'] is None


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
