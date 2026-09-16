from types import SimpleNamespace as NS
import pytest
from broker_runtime.prepare_olv_capacity import patch_book, prepare


def test_snapshot_exports_observed_remaining_and_filled_without_changing_total():
    source='''import json
def snap_account(t):
    try:
        o = t.order
        out = {"orders": []}
        out["orders"].append({
                "qty": _num(o.totalQuantity), "order_type": o.orderType,
        })
        # Today's executions.
        return out
    finally:
        pass
'''
    namespace={'_num':float}
    exec(compile(patch_book(source),'<book-candidate>','exec'),namespace)
    value=namespace['snap_account'](NS(order=NS(totalQuantity=100,orderType='LMT'),
        orderStatus=NS(remaining=40,filled=60)))
    assert value['orders']==[{'qty':100.,'order_type':'LMT','remaining':40.,'filled':60.}]
    assert value['orders_source_at'] > 0


def test_changed_source_fails_before_output_creation(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'book_snapshot.py').write_text('changed')
    output=tmp_path/'candidate'
    with pytest.raises(ValueError,match='source changed'):
        prepare(source,output)
    assert not output.exists()


def test_patcher_cannot_silently_accept_a_second_application():
    with pytest.raises(ValueError):patch_book('import json\n')
