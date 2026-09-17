"""Retired FOMC cannot return via legacy inputs, calibration or UI payloads."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import fragility_core as core

FOMC = "Pre-FOMC Rally"


def case():
    spy = pd.Series(100., index=pd.bdate_range('2025-01-02', periods=320))
    signals = {name: {'on': False, 'signal_history': pd.Series(False, index=spy.index)}
               for name in core.ACTIVE_RISK_SIGNALS}
    signals['Distribution Dominance']['signal_history'].iloc[260:300] = True
    signals['VIX Range Compression']['signal_history'].iloc[290:] = True
    signals['VIX Range Compression']['on'] = True
    stats = {'signals': {name: {'horizons': {h: {'diff_mean': -1.}
                         for h in core.HORIZON_DAYS}} for name in signals}}
    return spy, signals, stats


@pytest.mark.parametrize('fire_slice', [slice(0, 0), slice(270, 275), slice(318, 320)])
def test_legacy_fomc_neither_numerator_nor_denominator(fire_slice):
    spy, signals, stats = case()
    hist = pd.Series(False, index=spy.index)
    hist.iloc[fire_slice] = True
    legacy_signals = {**signals, FOMC: {'on': bool(hist.iloc[-1]), 'signal_history': hist}}
    legacy_stats = {'signals': {**stats['signals'], FOMC: {'horizons': {
        h: {'diff_mean': -999.} for h in core.HORIZON_DAYS}}}}
    pd.testing.assert_frame_equal(core.compute_fragility_timeseries(signals, spy, stats),
                                  core.compute_fragility_timeseries(legacy_signals, spy, legacy_stats))
    assert core.compute_horizon_fragility(signals, 1.1, stats, {'drawdown': 0}, spy) == \
        core.compute_horizon_fragility(legacy_signals, 1.1, legacy_stats, {'drawdown': 0}, spy)
    assert all(core._signal_edge(legacy_stats, FOMC, h) == 0 for h in core.HORIZON_DAYS)


def test_fomc_only_preserves_missing_data_semantics():
    spy, _, stats = case()
    pd.testing.assert_frame_equal(core.compute_fragility_timeseries({}, spy, stats),
        core.compute_fragility_timeseries({FOMC: {'signal_history': pd.Series(True, index=spy.index)}}, spy, stats))


def test_legacy_calibration_is_filtered_without_rewriting_file(tmp_path, monkeypatch):
    path = tmp_path/'stats.json'
    original = json.dumps({'signals': {FOMC: {'horizons': {}}, 'Dispersion': {'horizons': {}}}})
    path.write_text(original)
    monkeypatch.setattr(core, 'HORIZON_STATS_PATH', str(path))
    assert set(core.load_horizon_stats()['signals']) == {'Dispersion'}
    assert path.read_text() == original


def test_shadow_legacy_input_has_no_weight():
    from fragility_simple import compute_simple_dial, SIMPLE_SIGNALS
    spy, signals, _ = case()
    before = compute_simple_dial(signals, spy.index)
    after = compute_simple_dial({**signals, FOMC: {'signal_history': pd.Series(True, index=spy.index)}}, spy.index)
    pd.testing.assert_frame_equal(before, after)
    assert before.attrs['n_signals'] == after.attrs['n_signals']
    assert FOMC not in SIMPLE_SIGNALS and len(SIMPLE_SIGNALS) == 6


def test_trade_console_keeps_identical_classification_and_legacy_fingerprint():
    from scripts.build_trade_console_stats import (
        ABBR, build_config_frame, class_fingerprint, CLASS_SET_VERSION, CLASS_PRECEDENCE)
    spy, signals, _ = case()
    signals = {k: v for k, v in signals.items() if k in ABBR}
    old_signals = {**signals, FOMC: {'signal_history': pd.Series(True, index=spy.index)}}
    pd.testing.assert_frame_equal(build_config_frame(signals, spy), build_config_frame(old_signals, spy))
    old_basis = '|'.join(sorted(old_signals))+'||'+CLASS_SET_VERSION+'||'+'>'.join(CLASS_PRECEDENCE)
    old_fingerprint = hashlib.sha256(old_basis.encode()).hexdigest()[:16]
    assert class_fingerprint(signals) == class_fingerprint(old_signals) == old_fingerprint
    assert class_fingerprint(set(signals)-{'Dispersion'}) != old_fingerprint


def test_active_stats_generator_rejects_retired_signal():
    from scripts.build_signal_horizon_stats import signal_block
    spy, _, _ = case()
    assert signal_block(FOMC, pd.Series(True, index=spy.index), spy) is None


def test_daily_producer_returns_only_current_components(monkeypatch):
    import daily_risk_report as report
    spy, _, _ = case()
    def component(*args):
        return {'on': False, 'signal_history': pd.Series(False, index=spy.index)}
    for name in ['compute_da_signal', 'compute_vix_range_compression',
                 'compute_defensive_leadership', 'compute_low_ar_signal',
                 'compute_seasonal_divergence_signal', 'compute_dispersion_signal',
                 'compute_pc_complacency_signal']:
        monkeypatch.setattr(report, name, component)
    monkeypatch.setattr(core, 'load_horizon_stats', lambda: None)
    # Any accidental reintroduced call must fail instead of fetching data.
    def retired(*args):
        raise AssertionError('Retired component was evaluated')
    monkeypatch.setattr(report, 'compute_fomc_signal', retired, raising=False)
    result = report.compute_all_signals(spy.to_frame('Close'), pd.DataFrame(index=spy.index),
                                        pd.DataFrame(index=spy.index))
    assert set(result['signals_ordered']) == set(core.ACTIVE_RISK_SIGNALS)


def test_frontend_filters_legacy_payload_without_mutating_source():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser payload regression')
    script = Path(__file__).parents[1]/'site/assets/risk.js'
    probe = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={document:{addEventListener(){}}};vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const input={signals:[{name:'Pre-FOMC Rally',on:true},{name:'Dispersion',on:false}],
 signal_detail:{'Pre-FOMC Rally':{},Dispersion:{}},n_active:1,
 fragility:{'5d':99,'21d':50,'63d':30},fragility_10d:{'5d':99,'63d':29},
 atr_downside:{signals:{'Pre-FOMC Rally':{},Dispersion:{}}}};
const old=JSON.stringify(input),out=context.currentRiskPayload(input);
assert.equal(JSON.stringify(input),old);assert.equal(out.signals.length,1);
assert.equal(out.n_active,0);assert.equal(out.fragility['5d'],undefined);
assert.equal(out.fragility['21d'],undefined);assert.equal(out.fragility['63d'],30);
assert.equal(out.signal_detail['Pre-FOMC Rally'],undefined);
assert.equal(out.atr_downside.signals['Pre-FOMC Rally'],undefined);
const fresh={signals:[{name:'Dispersion',on:true}],fragility:{'5d':25,'21d':30,'63d':35}};
assert.equal(context.currentRiskPayload(fresh).fragility['5d'],25);
'''
    subprocess.run([node, '-e', probe, str(script)], check=True, capture_output=True, text=True)
