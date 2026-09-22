"""Run patched deployed functions with a fake broker, never import live code."""
import ast
import datetime as dt
import os
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from broker_runtime import auction_lifecycle, event_contract, prepare
from event_sleeve import EVENT_SLEEVE
from tests.spent_preparers import retired_prepare_auction, skip_prepare_auction


def row(trade="T2_FOMC_MIDTERM_SHORT", day="2026-09-16", entry=False):
    symbol, opening, closing, exit_type = event_contract.TRADES[trade]
    kind = "entry" if entry else "exit"
    ot = "MOC" if entry else exit_type
    return dict(Trade=trade, Ticker=symbol, Action=opening if entry else closing,
                Entry_Date="2026-09-10", Execute_On="2026-09-10" if entry else day,
                Execution_ID=f"{trade}|2026-09-10|{kind}", Quantity=100,
                Order_Type=ot, TIF="MOC" if ot == "MOC" else "OPG",
                Ref_Close=100., Ref_Notional=10000., Scan_Source="Event")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    skip_prepare_auction()
    source = Path(os.environ.get("IBKR_REVIEW_SOURCE", str(Path.home() / "OneDrive" / "trading_ibkr"))) / "event_moo.py"
    if not source.exists():
        pytest.skip("reviewed external source unavailable")
    text = prepare.patch_auction(source.read_text(encoding="utf-8-sig"), event=True)
    compile(text, "candidate-event_moo.py", "exec")
    funcs = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
             and n.name in {"_norm_today", "load_event_rows", "process_orders"}]
    monkeypatch.setitem(__import__("sys").modules, "auction_lifecycle", auction_lifecycle)
    monkeypatch.setitem(__import__("sys").modules, "event_contract", event_contract)
    monkeypatch.delenv("LEGEND_ETF_PRIMARY_ACCOUNT", raising=False)
    migration = tmp_path / "auction_intents" / "event_migration.json"
    migration.parent.mkdir()
    migration.write_text(json.dumps(dict(schema="event-migration.v1", account="TEST_PRIMARY",
                                         policy="hold_all_legacy_cycles", through_entry_date="2026-09-09")))
    placed = []
    broker = NS(connect=lambda *a, **k: None, disconnect=lambda: None,
                sleep=lambda _: None, managedAccounts=lambda: ["TEST_PRIMARY"],
                reqAllOpenOrders=lambda: None, reqExecutions=lambda: None,
                openTrades=lambda: [], fills=lambda: [],
                positions=lambda: [NS(account="TEST_PRIMARY", position=1000,
                                      contract=NS(symbol=t, secType="STK", currency="USD"))
                                   for t in ["SPY", "IWM", "SVXY"]],
                qualifyContracts=lambda c: [c])
    def place(ib, contract, order, **kwargs):
        placed.append(order)
        order.orderId = 1
        return NS(order=order, orderStatus=NS(status="Submitted"))
    env = dict(pd=pd, dt=dt, __file__=str(tmp_path / "event_moo.py"),
               IB=lambda: broker, Stock=lambda *a: NS(conId=42), Order=NS,
               PRIMARY_IP="fixture", PRIMARY_PORT=0, PRIMARY_CLIENT_ID=147,
               OPG_CUTOFF=dt.time(9, 25), MOC_CUTOFF=dt.time(15, 30),
               signal_ref=lambda *v: "|".join(v), order_fingerprint=lambda *v: v,
               load_placed_today=lambda _: (set(), set()), journal_placed=lambda *a: None,
               IB_ACTION={"BUY": "BUY", "SELL": "SELL", "SELL_SHORT": "SELL", "BUY_TO_COVER": "BUY"},
               TERMINAL_REJECT_STATUSES={"Cancelled", "Inactive", "ApiCancelled"},
               guarded_place_order=place,
               guarded_cancel_order=lambda *a: pytest.fail("unexpected cancellation"),
               EVENT_TAB_NAME="Event", EVENT_TRADES=set(EVENT_SLEEVE),
               EVENT_UNIVERSE={"SPY", "IWM", "SVXY"}, MAX_ROWS=3,
               MAX_TOTAL_REF_NOTIONAL=350000, VALID_ACTIONS={"BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"},
               EventSheetError=ValueError, gspread=NS(WorksheetNotFound=LookupError))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "candidate-functions", "exec"), env)
    return env, placed


@pytest.mark.parametrize("trade", EVENT_SLEEVE)
def test_cross_day_retry_submits_once_for_every_strategy(runner, trade):
    env, placed = runner
    for day in ["2026-09-16", "2026-09-17"]:
        result = env["process_orders"](pd.DataFrame([row(trade, day)]), day,
                                        now=dt.datetime.fromisoformat(day + "T09:05:00"))
    assert len(placed) == 1
    assert placed[0].orderRef.endswith("|2026-09-10")
    assert placed[0].account == "TEST_PRIMARY"
    assert result[0]["Status"] == "PENDING_RECONCILIATION"


def test_uncertain_delivery_is_never_retried(runner):
    env, placed = runner
    def uncertain(*args, **kwargs):
        placed.append(args[2])
        raise TimeoutError("delivery unknown")
    env["guarded_place_order"] = uncertain
    for day in ["2026-09-16", "2026-09-17"]:
        result = env["process_orders"](pd.DataFrame([row(day=day)]), day,
                                        now=dt.datetime.fromisoformat(day + "T09:05:00"))
    assert len(placed) == 1 and result[0]["Status"] == "PENDING_RECONCILIATION"


def test_pending_ack_is_not_cancelled_or_success(runner):
    env, _ = runner
    env["guarded_place_order"] = lambda *a, **k: NS(order=a[2], orderStatus=NS(status="PendingSubmit"))
    result = env["process_orders"](pd.DataFrame([row()]), "2026-09-16", now=dt.datetime(2026, 9, 16, 9, 5))
    assert result[0]["Status"] == "PENDING_RECONCILIATION"


def test_missed_auction_can_be_retried_without_market_fallback(runner):
    env, placed = runner
    result = env["process_orders"](pd.DataFrame([row()]), "2026-09-16", now=dt.datetime(2026, 9, 16, 9, 26))
    assert not placed and result[0]["Status"] == "MISSED_OPG_CUTOFF"
    env["process_orders"](pd.DataFrame([row(day="2026-09-17")]), "2026-09-17", now=dt.datetime(2026, 9, 17, 9, 5))
    assert len(placed) == 1 and placed[0].tif == "OPG"


def test_stale_exit_tab_is_not_replayed(runner):
    env, _ = runner
    sheet = NS(worksheet=lambda _: NS(get_all_records=lambda: [row()]))
    assert env["load_event_rows"](sheet, "2026-09-17").empty


@pytest.mark.parametrize("change", [dict(Entry_Date=""), dict(Entry_Date=float("nan")),
                                  dict(Execution_ID="wrong"), dict(Ticker="IWM"),
                                  dict(Action="SELL"), dict(Quantity=0.5),
                                  dict(Ref_Close=float("inf")), dict(Order_Type="MOC", TIF="MOC")])
def test_invalid_identity_or_strategy_rejected_before_broker(runner, change):
    env, placed = runner
    sheet = NS(worksheet=lambda _: NS(get_all_records=lambda: [{**row(), **change}]))
    with pytest.raises(ValueError):
        env["load_event_rows"](sheet, "2026-09-16")
    assert placed == []


def test_execution_contract_matches_all_producer_strategies():
    assert set(event_contract.TRADES) == set(EVENT_SLEEVE)
    for trade, cfg in EVENT_SLEEVE.items():
        symbol, opening, _, _ = event_contract.TRADES[trade]
        assert symbol == cfg["ticker"]
        assert opening == ("BUY" if cfg["side"] == "LONG" else "SELL_SHORT")


@retired_prepare_auction
def test_event_only_candidate_checks_source_and_avoids_unrelated_runners(tmp_path):
    source = Path(os.environ.get("IBKR_REVIEW_SOURCE", str(Path.home() / "OneDrive" / "trading_ibkr")))
    if not (source / "event_moo.py").exists():
        pytest.skip("reviewed external source unavailable")
    output = tmp_path / "candidate"
    assert prepare.prepare(source, output, event_only=True) == ["event_moo.py"]
    assert {p.name for p in output.iterdir()} == {"event_moo.py", "auction_lifecycle.py", "event_contract.py"}
    for file in output.glob("*.py"):
        compile(file.read_text(), file.name, "exec")


def test_changed_quantity_on_retry_requires_reconciliation(runner):
    env, placed = runner
    env["process_orders"](pd.DataFrame([row()]), "2026-09-16", now=dt.datetime(2026, 9, 16, 9, 5))
    result = env["process_orders"](pd.DataFrame([{**row(day="2026-09-17"), "Quantity": 60}]),
                                   "2026-09-17", now=dt.datetime(2026, 9, 17, 9, 5))
    assert len(placed) == 1 and result[0]["Status"] == "PLACE_FAILED"


def test_rejected_primary_does_not_cancel_other_account_order(runner):
    env, placed = runner
    broker = env["IB"]()
    other = NS(order=NS(account="TEST_OTHER", orderRef="SPY|BUY_TO_COVER|T2_FOMC_MIDTERM_SHORT|2026-09-10"))
    broker.openTrades = lambda: [other]
    def reject(*args, **kwargs):
        placed.append(args[2])
        return NS(order=args[2], orderStatus=NS(status="Cancelled"))
    env["guarded_place_order"] = reject
    result = env["process_orders"](pd.DataFrame([row()]), "2026-09-16", now=dt.datetime(2026, 9, 16, 9, 5))
    assert len(placed) == 1 and result[0]["Status"] == "PENDING_RECONCILIATION"
    # fixture's cancel function fails the test on ANY cancellation.


@pytest.mark.parametrize("trade,cutoff", [("T2_FOMC_MIDTERM_SHORT", dt.time(9,25)),
                                         ("V4_POSTOPEX_VOL", dt.time(15,30))])
@pytest.mark.parametrize("delay_at", ["qualification", "guard_callback", "guard_direct"])
def test_cutoff_checked_after_slow_reads_and_at_wire(runner, trade, cutoff, delay_at):
    env, placed = runner
    broker = env["IB"]()
    late = dt.datetime.combine(dt.date(2026, 9, 16), cutoff)
    clock = [late - dt.timedelta(seconds=1)]
    broker.placeOrder = lambda contract, order: placed.append(order)
    if delay_at == "qualification":
        def qualify(contract):
            clock[0] = late
            return [contract]
        broker.qualifyContracts = qualify
    def guard(client, contract, order, **kwargs):
        if delay_at != "qualification":
            clock[0] = late
        if delay_at == "guard_callback":
            kwargs["before_broker_call"]()
        return client.placeOrder(contract, order)
    env["guarded_place_order"] = guard
    result = env["process_orders"](pd.DataFrame([row(trade)]), "2026-09-16", clock=lambda: clock[0])
    assert placed == [] and result[0]["Status"] == "PLACE_FAILED"


@pytest.mark.parametrize("date,allowed", [("2026-09-16", False), ("2026-09-17", False), ("2026-09-18", True)])
def test_migration_fence_boundary(tmp_path, date, allowed):
    (tmp_path / "event_migration.json").write_text(json.dumps(dict(
        schema="event-migration.v1", account="TEST_PRIMARY", policy="hold_all_legacy_cycles",
        through_entry_date="2026-09-17")))
    record = row(entry=True)
    record.update(Entry_Date=date, Execute_On=date,
                  Execution_ID=f"T2_FOMC_MIDTERM_SHORT|{date}|entry")
    if allowed:
        event_contract.assert_migrated_cycle(tmp_path, "TEST_PRIMARY", record)
    else:
        with pytest.raises(RuntimeError, match="legacy"):
            event_contract.assert_migrated_cycle(tmp_path, "TEST_PRIMARY", record)


@pytest.mark.parametrize("body", [None, "{broken", '{}', json.dumps(dict(
    schema="event-migration.v1", account="OTHER", policy="hold_all_legacy_cycles", through_entry_date="2026-09-09"))])
def test_missing_or_invalid_migration_never_submits(runner, body):
    env, placed = runner
    # Point at a fresh directory rather than deleting a receipt.
    folder = Path(env["__file__"]).parent / "without_valid_migration"
    env["__file__"] = str(folder / "event_moo.py")
    if body is not None:
        (folder / "auction_intents").mkdir(parents=True)
        (folder / "auction_intents" / "event_migration.json").write_text(body)
    result = env["process_orders"](pd.DataFrame([row()]), "2026-09-16", now=dt.datetime(2026, 9, 16, 9, 5))
    assert placed == [] and result[0]["Status"] == "PLACE_FAILED"
