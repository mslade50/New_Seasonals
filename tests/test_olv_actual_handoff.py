"""Actual-tranche OLV candidate tests; no deployed module imports or broker I/O."""
import ast
import datetime as dt
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from broker_runtime import olv_contract as contract
from broker_runtime import prepare

STRATEGY = "Oversold Low Volume"
ENTRY_REF = f"SPY|BUY|{STRATEGY}|2026-09-01"
SOURCE = Path(os.environ.get("IBKR_REVIEW_SOURCE", "C:/Users/McKinley Slade/OneDrive/trading_ibkr"))


def row(**changes):
    return dict(Symbol="SPY", Quantity=80, Time_Exit_Date="2026-09-16", Execute_On="2026-09-04",
                Strategy_Ref=STRATEGY, account_key="primary", broker_account="TEST_PRIMARY",
                con_id=42, tranche_id="fixture", ref_date="2026-09-01", entry_order_ref=ENTRY_REF, **changes)


def trade(con_id=42, account="TEST_PRIMARY", entry_ref=ENTRY_REF, qty=80):
    return NS(contract=NS(conId=con_id, symbol="SPY"),
              order=NS(account=account, orderRef=entry_ref, clientId=9, orderId=7, permId=701,
                       totalQuantity=qty, goodAfterTime="20260916 15:59:00 US/Eastern", action="SELL", orderType="MKT"),
              orderStatus=NS(filled=0, remaining=qty))


def fill(exec_id="fixture.01", qty=30, side="SLD", account="TEST_PRIMARY", con_id=42, entry_ref=ENTRY_REF):
    return NS(contract=NS(conId=con_id), execution=NS(execId=exec_id, shares=qty, side=side,
              acctNumber=account, orderRef=entry_ref))


class Broker:
    def __init__(self, fills=()): self.rows = list(fills)
    def reqExecutions(self): return self.rows
    def fills(self): return self.rows
    def sleep(self, _): pass


def test_overdue_rows_keep_their_original_identity_for_next_auction():
    staged = contract.validate_rows(pd.DataFrame([row()]), "2026-09-08", STRATEGY)
    assert len(staged) == 1 and staged.iloc[0]["Execute_On"] == "2026-09-04"
    assert staged.iloc[0]["entry_order_ref"] == ENTRY_REF


@pytest.mark.parametrize("changes", [{"account_key": "pa"}, {"broker_account": ""}, {"Quantity": 1.5},
    {"con_id": 42.2}, {"tranche_id": ""}, {"entry_order_ref": ENTRY_REF.replace("09-01", "09-02")}])
def test_ambiguous_rows_fail_before_runner_connection(changes):
    staged = row(); staged.update(changes)
    with pytest.raises(ValueError):
        contract.validate_rows(pd.DataFrame([staged]), "2026-09-08", STRATEGY)


def test_two_tranches_cannot_claim_one_bracket():
    second = row(); second["tranche_id"] = "different-tranche"
    with pytest.raises(ValueError, match="same bracket"):
        contract.validate_rows(pd.DataFrame([row(), second]), "2026-09-08", STRATEGY)


def test_exact_matching_excludes_other_account_contract_or_reference():
    good = trade()
    candidates = [trade(account="TEST_OTHER"), trade(con_id=43), trade(entry_ref=ENTRY_REF + "other"), good]
    assert contract.matching_time_legs(candidates, row()) == [good]
    bound = row(); bound["source_time_perm_id"] = 702
    assert contract.matching_time_legs(candidates, bound) == []


def test_corrections_deduplicate_before_contract_or_side_attribution():
    broker = Broker([fill(qty=100), fill("fixture.02", qty=90), fill("other.01", account="TEST_OTHER")])
    assert contract.sold_for_entry(broker, "TEST_PRIMARY", 42, ENTRY_REF) == 90
    broker.rows.append(fill("fixture.03", con_id=43))
    assert contract.sold_for_entry(broker, "TEST_PRIMARY", 42, ENTRY_REF) == 0
    broker.rows.append(fill("fixture.04", side="BOT"))
    assert contract.sold_for_entry(broker, "TEST_PRIMARY", 42, ENTRY_REF) == 0


def test_rearm_cannot_turn_a_past_or_missing_deadline_into_market_now():
    now = dt.datetime(2026, 9, 8, 14, tzinfo=dt.timezone.utc)
    assert not contract.future_time_exit("", now)
    assert not contract.future_time_exit("20260908 09:59:00 US/Eastern", now)
    assert contract.future_time_exit("20260916 15:59:00 US/Eastern", now)


@pytest.fixture
def patched(monkeypatch):
    if not (SOURCE / "olv_exit_moo.py").exists():
        pytest.skip("reviewed external source unavailable; portable handoff tests still run")
    source = prepare.patch_olv((SOURCE / "olv_exit_moo.py").read_text(encoding="utf-8-sig"))
    monkeypatch.setitem(__import__("sys").modules, "olv_contract", contract)
    return ast.parse(source)


def extract(tree, name, namespace):
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<prepared-olv-function>", "exec"), namespace)
    return namespace[name]


def base_namespace():
    return {"_row": lambda label, symbol, status, detail="": {"Status": status, "Detail": detail},
            "_now_time": lambda: dt.time(9, 0), "OPG_CUTOFF": dt.time(9, 25)}


def test_runner_pa_and_late_auction_stop_before_connection(patched):
    ns = base_namespace()
    ns["connect_account"] = lambda *a: pytest.fail("unexpected broker connection")
    function = extract(patched, "process_account", ns)
    rows = pd.DataFrame([row()])
    assert function("PA", "test", 0, 9, rows, "2026-09-08")[0]["Status"] == "CRITICAL_PRIMARY_ONLY"
    ns["_now_time"] = lambda: dt.time(9, 26)
    assert function("PRIMARY", "test", 0, 9, rows, "2026-09-08")[0]["Status"] == "CRITICAL_MISSED_AUCTION_PENDING"


def test_no_nearest_date_or_ambiguous_same_date_bracket(patched):
    pick = extract(patched, "pick_time_leg", {"_gat8": lambda value: value.order.goodAfterTime[:8]})
    assert pick([trade()], "2026-09-15") is None
    assert pick([trade(), trade()], "2026-09-16") is None


def test_staged_actual_quantity_mismatch_preserves_bracket(patched):
    working = trade(qty=100)
    ns = base_namespace()
    ns.update(STRATEGY_NAME=STRATEGY, load_placed=lambda _: {}, _fresh_open_trades=lambda _: [working],
        _order_ref=lambda value: value.order.orderRef, _trade_account=lambda value: value.order.account,
        _is_active=lambda _: True, _ref_matches=lambda ref, base: ref.startswith(base),
        _journal_rows_for=lambda *a: [], pick_journaled_leg=lambda *a: (None, False),
        pick_time_leg=lambda *a: working, retry_decision=lambda _: ("FRESH", 1),
        signal_ref=lambda *a: "|".join(a), _cancel_legs=lambda *a: pytest.fail("bracket mutated"))
    result = extract(patched, "process_account", ns)("PRIMARY", "test", 0, 9, pd.DataFrame([row()]),
        "2026-09-08", connected={"ib": Broker(), "account": "TEST_PRIMARY"})
    assert result[0]["Status"] == "CRITICAL_TAGGED_QUANTITY_MISMATCH"


def execute_fixture(patched, *, readable=True, held=180, broker=None, place_error=False):
    calls = []
    ns = base_namespace()
    def place(ib, contract, order, **kwargs):
        calls.append(order.totalQuantity)
        if place_error:
            raise TimeoutError("ambiguous send")
        order.clientId, order.orderId, order.permId = 9, 7, 701
        return NS(order=order)
    ns.update(_held_now=lambda *a: (held, readable), _journal_quiet=lambda *a, **k: True,
        journal_placed=lambda *a, **k: None, _poll_status=lambda *a: "Submitted", Order=NS,
        guarded_place_order=place, STATE_PREPARED="PREPARED", STATE_ACKNOWLEDGED="ACKNOWLEDGED",
        STATE_UNKNOWN="UNKNOWN", STATE_SKIPPED_FLAT="SKIPPED_FLAT", ACK_STATUSES={"Submitted", "Filled"},
        ExitJournalError=RuntimeError)
    function = extract(patched, "_execute_exit", ns)
    result = function(broker or Broker([fill()]), "PRIMARY", "TEST_PRIMARY", "SPY", NS(conId=42),
        "exit-ref", "exit-ref", 80, 1, "20260916 15:59:00 US/Eastern", ENTRY_REF, "2026-09-08",
        {"con_id": 42, "tranche_sold_before": 20}, [], queued_before=0)
    return result, calls


def test_fill_during_cancel_reduces_owned_quantity_even_when_other_strategy_keeps_net_long(patched):
    result, calls = execute_fixture(patched)
    assert result["Status"] == "SENT_OPG" and calls == [70]


@pytest.mark.parametrize("kwargs,status", [({"readable": False}, "CRITICAL_POSITION_RECONCILE_AFTER_CANCEL"),
    ({"held": 60}, "CRITICAL_TAGGED_POSITION_MISMATCH_AFTER_CANCEL")])
def test_unknown_or_inconsistent_position_after_cancel_never_sends(patched, kwargs, status):
    result, calls = execute_fixture(patched, **kwargs)
    assert result["Status"] == status and calls == []


def test_delivery_exception_has_one_send_and_no_replacement(patched):
    result, calls = execute_fixture(patched, place_error=True)
    assert result["Status"] == "CRITICAL_EXIT_DELIVERY_UNKNOWN" and calls == [70]


def test_main_only_has_primary_endpoint(patched):
    main = next(node for node in patched.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    specs = next(node.value for node in ast.walk(main) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "specs" for target in node.targets))
    assert [entry.elts[0].value for entry in specs.elts] == ["PRIMARY"]
