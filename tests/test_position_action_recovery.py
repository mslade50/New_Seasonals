import copy
import hashlib
import json

import pytest

from broker_runtime import position_actions as actions
from broker_runtime.position_action_recovery import abandon, reviewed_abort


def evidence():
    leg = dict(identity=["PRIMARY", 42, 7, 1, 101], qty=100, filled_before=0,
               source_key="perm:101", order_type="LMT", lmt=120, tif="GTC")
    record = dict(version=1, phase="attention", mutation="resize exit", id="fixture",
                  payload=dict(_broker_account="PRIMARY", con_id=42, readd=False),
                  legs=[leg], held=100, quantity=20, closing="SELL", removed=[],
                  error="exit resize lacks broker acknowledgement")
    order = dict(account="PRIMARY", con_id=42, client_id=7, order_id=1, perm_id=101,
                 action="SELL", parent_id=0, status="Submitted", qty=80, remaining=80,
                 filled=0, order_type="LMT", lmt=120, tif="GTC")
    account = dict(broker_account="PRIMARY", orders_source_at=990, fills_complete=True,
                   fills_source_at=990000, orders=[order])
    return record, account


def test_abort_preserves_partial_edits_and_original_receipt(tmp_path):
    record, account = evidence()
    actions.save(tmp_path, record)
    path = actions.record_path(tmp_path, record["id"])
    original = path.read_bytes()
    result = abandon(tmp_path, record["id"], hashlib.sha256(original).hexdigest(),
                     account, now=1000, reason="Operator will finish exit edits",
                     backup_dir=tmp_path / "backup")
    assert result["phase"] == "done" and result["result"]["ok"] is False
    assert result["result"]["state"] == "rejected"
    assert result["error"] == record["error"]
    assert json.loads(path.read_bytes()) == result
    assert next((tmp_path / "backup").glob("*.json")).read_bytes() == original
    assert account["orders"][0]["qty"] == 80


@pytest.mark.parametrize("change", [
    lambda r, a: r.update(wire=[]),
    lambda r, a: r.update(phase="pending"),
    lambda r, a: r.update(mutation="submit close"),
    lambda r, a: r.update(removed=["perm:101"]),
    lambda r, a: r.update(add_context={}),
    lambda r, a: r["payload"].update(readd=True),
    lambda r, a: a.update(broker_account="PA"),
    lambda r, a: a.update(orders_source_at=900),
    lambda r, a: a.update(fills_complete=False),
    lambda r, a: a.update(fills_source_at=900000),
    lambda r, a: a.update(orders=[]),
    lambda r, a: a["orders"][0].update(perm_id=102),
    lambda r, a: a["orders"][0].update(filled=1),
    lambda r, a: a["orders"][0].update(status="PendingSubmit"),
    lambda r, a: a["orders"][0].update(qty=81, remaining=81),
    lambda r, a: a["orders"][0].update(lmt=121),
])
def test_ambiguous_or_changed_operations_stay_blocked(change):
    record, account = evidence()
    change(record, account)
    original = copy.deepcopy(record)
    with pytest.raises(ValueError):
        reviewed_abort(record, account, now=1000, reason="Reviewed")
    assert record == original


def test_reviewed_hash_must_still_match(tmp_path):
    record, account = evidence()
    actions.save(tmp_path, record)
    with pytest.raises(ValueError, match="changed since review"):
        abandon(tmp_path, record["id"], "wrong", account, now=1000,
                reason="Reviewed", backup_dir=tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_completed_parent_reference_is_allowed():
    record, account = evidence()
    account["orders"][0]["parent_id"] = 99
    assert reviewed_abort(record, account, now=1000, reason="Reviewed")["phase"] == "done"


def test_working_entry_parent_is_rejected():
    record, account = evidence()
    account["orders"][0]["parent_id"] = 99
    account["orders"].append(dict(con_id=42, order_id=99, client_id=7))
    with pytest.raises(ValueError):
        reviewed_abort(record, account, now=1000, reason="Reviewed")


def test_manual_edit_unblocks_but_original_command_is_never_replayed(tmp_path, monkeypatch):
    from broker_runtime import order_mutations
    record, account = evidence()
    actions.save(tmp_path, record)
    calls = []
    def mutate(*args, **kwargs):
        calls.append(args[2])
        return dict(ok=True, state="executed", detail="Fixture acknowledgement")
    monkeypatch.setattr(order_mutations.life, "mutate_one", mutate)
    ns = dict(POSITION_ACTION_STATE_DIR=tmp_path,
              _out=lambda ok, state, detail: dict(ok=ok, state=state, detail=detail))
    payload = dict(_command_id="edit", _broker_account="PRIMARY", con_id=42,
                   client_id=7, order_id=1, perm_id=101, qty=80)
    blocked = order_mutations.run(ns, None, payload, "", 0, 7, modify=True)
    assert blocked["state"] == "rejected" and "earlier position action" in blocked["detail"]
    assert not calls
    original = actions.record_path(tmp_path, "fixture").read_bytes()
    resolved = abandon(tmp_path, "fixture", hashlib.sha256(original).hexdigest(),
                       account, now=1000, reason="Operator completing exit edits",
                       backup_dir=tmp_path / "backup")
    result = order_mutations.run(ns, None, payload, "", 0, 7, modify=True)
    assert result["ok"] and calls == [payload]
    # The old close remains terminal and rejected, never pending for agent resume.
    assert resolved["phase"] == "done" and resolved["result"]["state"] == "rejected"
    assert next(actions.records(tmp_path))["result"] == resolved["result"]
