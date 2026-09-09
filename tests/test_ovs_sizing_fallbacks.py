"""Pure tests of generated staging functions, never import the broker runtime."""
import ast
import hashlib
import math
from pathlib import Path

import pytest

from broker_runtime.prepare_ovs_sizing import SOURCE_SHA256, defaults, patch_staging, prepare


def fixture_source():
    return '''import math
OVS_PATH2_QTY_MULT = 0.15
OVS_PATH2_DAILY_CAP_PCT_DEFAULT = 1.0

def _ovs_path2_mult(row):
    return OVS_PATH2_QTY_MULT

def main(df, p2_mask):
    if True:
        if True:
            cap_pct = OVS_PATH2_DAILY_CAP_PCT_DEFAULT
            try:
                stamped = pd.to_numeric(
                    df.loc[p2_mask, 'Path2_Daily_Cap_Pct'], errors='coerce').dropna()
                if not stamped.empty and float(stamped.iloc[0]) > 0:
                    cap_pct = float(stamped.iloc[0])
            except (KeyError, ValueError, TypeError):
                pass
            return cap_pct
'''


def pure_helpers(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) or
             isinstance(node, ast.FunctionDef) and node.name.startswith('_ovs_')]
    env = {'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<staging helpers only>', 'exec'), env)
    return env


@pytest.fixture
def helpers():
    return pure_helpers(patch_staging(fixture_source()))


def test_defaults_come_from_scaled_config():
    assert defaults() == {'path2_multiplier': .2, 'path2_daily_cap_pct': 1.125}


@pytest.mark.parametrize('bad', [None, '', 'bad', 0, -1, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['Path1_Bps', 'Path2_Bps'])
def test_invalid_or_partial_pair_uses_configured_fallback(helpers, capsys, bad, field):
    row = {'Path1_Bps': 60, 'Path2_Bps': 12, field: bad}
    assert helpers['_ovs_path2_mult'](row) == .2
    assert '[WARN]' in capsys.readouterr().out


def test_missing_pair_and_valid_custom_pair(helpers, capsys):
    assert helpers['_ovs_path2_mult']({}) == .2
    capsys.readouterr()
    assert helpers['_ovs_path2_mult']({'Path1_Bps':'50','Path2_Bps':'5'}) == .1
    assert capsys.readouterr().out == ''


def test_overflow_ratio_falls_back(helpers, capsys):
    assert helpers['_ovs_path2_mult']({'Path1_Bps':1e-300,'Path2_Bps':1e300}) == .2
    assert '[WARN]' in capsys.readouterr().out


@pytest.mark.parametrize('values', [[], [None], ['bad', 0, -1, float('nan'), float('inf')]])
def test_missing_or_invalid_cap_warns(helpers, capsys, values):
    assert helpers['_ovs_path2_cap_pct'](values) == 1.125
    assert '[WARN]' in capsys.readouterr().out


def test_valid_cap_is_preserved(helpers, capsys):
    assert helpers['_ovs_path2_cap_pct']([None, '1.4', 1.5]) == 1.4
    assert capsys.readouterr().out == ''


def test_prepare_refuses_unreviewed_source_before_writing(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'order_staging.py').write_text(fixture_source())
    output = tmp_path / 'candidate'
    with pytest.raises(ValueError, match='changed'):
        prepare(source, output)
    assert not output.exists()


def test_installed_candidate_changes_only_reviewed_ovs_sections():
    path = Path('C:/Users/McKinley Slade/OneDrive/trading_ibkr/order_staging.py')
    if not path.exists():
        pytest.skip('reviewed external source is not installed on this host')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    original = ast.parse(raw.decode('utf-8-sig'))
    patched = ast.parse(patch_staging(raw.decode('utf-8-sig')))
    old_functions = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n) for n in patched.body if isinstance(n, ast.FunctionDef)}
    changed = {name for name in old_functions if old_functions[name] != new_functions[name]}
    assert changed == {'_ovs_path2_mult', 'pull_and_stage_orders'}
    assert new_functions.keys() - old_functions.keys() == {'_ovs_path2_cap_pct'}
    # Everything outside the two changed functions and fallback constants is identical.
    def unchanged(tree):
        return [ast.dump(n) for n in tree.body if not (
            isinstance(n, ast.FunctionDef) and n.name in ('_ovs_path2_mult','_ovs_path2_cap_pct','pull_and_stage_orders')
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ('OVS_PATH2_QTY_MULT','OVS_PATH2_DAILY_CAP_PCT_DEFAULT') for t in n.targets))]
    assert unchanged(original) == unchanged(patched)
