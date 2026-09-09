import ast,os
from pathlib import Path
import pytest
from broker_runtime.prepare_olv_production import prepare


def test_candidate_keeps_pa_logic_and_separates_primary_contract(tmp_path):
    source=Path(os.environ.get('OLV_PRODUCTION_REVIEW_SOURCE','C:/Users/McKinley Slade/OneDrive/trading_ibkr'))
    if not (source/'book_snapshot.py').exists():pytest.skip('reviewed broker sources unavailable')
    target=tmp_path/'candidate';prepare(source,target)
    original=ast.parse((source/'olv_exit_moo.py').read_text(encoding='utf-8-sig'))
    pa=ast.parse((target/'olv_exit_pa_legacy.py').read_text(encoding='utf-8'))
    def functions(tree):return {n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    old,new=functions(original),functions(pa)
    for name in old:
        if name!='main':assert old[name]==new[name],name
    primary=(target/'olv_exit_primary.py').read_text()
    assert "OLV_EXITS_TAB_NAME = 'OLV_Exits_Primary'" in primary
    assert "'olv_exit_primary_placed.json'" in primary
    assert "('PA', PA_IP" not in primary
    staged=(target/'order_staging.py').read_text()
    assert 'primary_df = primary_olv_deadlines(primary_df)' in staged
    assert 'df2 = build_pa_frame(final_df, scale)' in staged


def test_changed_installed_source_refuses_preparation(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'book_snapshot.py').write_text('changed')
    destination=tmp_path/'candidate'
    with pytest.raises(ValueError,match='source changed'):prepare(source,destination)
    assert not destination.exists()
