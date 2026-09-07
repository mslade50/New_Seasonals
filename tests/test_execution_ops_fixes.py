from __future__ import annotations

import datetime as dt
import io
import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

import daily_execution_report as report
from scripts import harvest_fills as harvest
from scripts.producer_health import inspect_health
from sleeve_fills import reconcile_event_fills


def fill(exec_id="sample.01", qty=100, account="TEST_PRIMARY", **extra):
    return {"exec_id": exec_id, "qty": qty, "account": account,
            "account_key": "primary", "time": "2026-09-04T20:00:00+00:00",
            "symbol": "SPY", "con_id": 42, "side": "BOT", **extra}


def test_correction_replaces_quantity_and_preserves_other_account():
    old = harvest.normalize([fill(), fill(account="TEST_OTHER")])
    new = harvest.normalize([fill("sample.02", 90)])
    result, _ = harvest.merge_fills(old, new)
    assert len(result) == 2
    assert result.loc[result.account.eq("TEST_PRIMARY"), "qty"].sum() == 90
    assert result.loc[result.account.eq("TEST_OTHER"), "qty"].sum() == 100
    old_revision, _ = harvest.merge_fills(result, old)
    assert old_revision.loc[old_revision.account.eq("TEST_PRIMARY"), "qty"].sum() == 90


def test_canonical_read_outage_never_uses_local_fallback(monkeypatch):
    import cache_io
    class Client:
        def get_object(self, **kwargs):
            raise TimeoutError("fixture outage")
    monkeypatch.setattr(cache_io, "_client", lambda: Client())
    monkeypatch.setattr(cache_io, "_r2_creds", lambda: {"R2_BUCKET": "fixture"})
    with pytest.raises(RuntimeError, match="preserved"):
        harvest.load_existing()


def test_canonical_write_uses_generation_and_original_etag(monkeypatch):
    import cache_io
    writes = []
    original = harvest.normalize([fill()])
    old_bytes = original.to_parquet(index=False)
    original.attrs.update(canonical_loaded=True, canonical_bytes=old_bytes, canonical_etag='"old"')
    merged, _ = harvest.merge_fills(original, harvest.normalize([fill("sample.02", 90)]))
    monkeypatch.setattr(cache_io, "_client", lambda: NS(put_object=lambda **kw: writes.append(kw)))
    monkeypatch.setattr(cache_io, "_r2_creds", lambda: {"R2_BUCKET": "fixture"})
    harvest.publish_canonical(merged, original)
    assert len(writes) == 3
    assert all(row["IfNoneMatch"] == "*" for row in writes[:2])
    assert writes[-1]["IfMatch"] == '"old"'
    assert pd.read_parquet(io.BytesIO(writes[-1]["Body"])).qty.sum() == 90


def test_primary_receipt_is_required_but_pa_failure_does_not_veto():
    now = "2026-09-06T12:00:00+00:00"
    payload = {"completeness": {"complete": False, "truncated": False,
               "accounts": {"primary": {"complete": True, "received_at": now, "source_at": now},
                            "pa": {"complete": False, "error": "offline"}}}}
    harvest.validate_source_completeness(payload, now=now)
    payload["completeness"]["accounts"]["primary"]["complete"] = False
    with pytest.raises(RuntimeError, match="Primary"):
        harvest.validate_source_completeness(payload, now=now)


@pytest.mark.parametrize("extra", [{"truncated": True}, {"incomplete_days": ["2026-09-04"]}, {"merge_error": "failed"}])
def test_incomplete_history_never_attests_primary(extra):
    now = "2026-09-06T12:00:00+00:00"
    payload = {"completeness": {**extra, "accounts": {"primary": {"complete": True, "received_at": now, "source_at": now}}}}
    with pytest.raises(RuntimeError):
        harvest.validate_source_completeness(payload, now=now)


def test_report_exit_identity_excludes_other_month_account_and_pending_entry():
    pos = {"con_id": 42, "account": "TEST_PRIMARY", "position": 100, "symbol": "MES", "sec_type": "FUT"}
    valid = {"con_id": 42, "account": "TEST_PRIMARY", "action": "SELL", "status": "Submitted", "client_id": 2, "order_id": 7, "parent_id": 0}
    parent = dict(valid, action="BUY", order_id=10)
    pending_child = dict(valid, parent_id=10, order_id=11)
    wrong_month = dict(valid, con_id=43)
    wrong_account = dict(valid, account="TEST_OTHER")
    assert report.exit_legs_for(pos, [valid, parent, pending_child, wrong_month, wrong_account]) == [valid]


def test_stale_book_is_not_a_current_report():
    now = dt.datetime.now(dt.timezone.utc)
    book = {"at": (now.timestamp() - 600) * 1000, "accounts": [{"key": "primary", "positions": [], "orders": []}]}
    with pytest.raises(RuntimeError, match="stale"):
        report.validate_book(book, now)


def test_no_send_applies_to_snapshot_failure(monkeypatch):
    monkeypatch.setenv("STATUS_TOKEN", "test-not-a-secret")
    monkeypatch.setattr(report, "fetch_book", lambda *a: (_ for _ in ()).throw(RuntimeError("fixture")))
    monkeypatch.setattr(report, "send_email", lambda *a, **k: pytest.fail("--no-send attempted mail"))
    monkeypatch.setattr("sys.argv", ["daily_execution_report.py", "--no-send", "--force"])
    assert report.main() == 1


def test_event_obligation_clears_only_after_attributed_exit():
    config = {"Event Test": {"ticker": "SPY", "side": "LONG"}}
    state = {"positions": {"Event Test": {"shares": 100, "entry_date": "2026-09-01", "status": "exit_pending"}}}
    entry = fill(order_ref="SPY|BUY|Event Test|2026-09-01")
    partial = fill("exit.01", 40, side="SLD", order_ref="SPY|SELL|Event Test|2026-09-01")
    reconcile_event_fills(state, harvest.normalize([entry, partial]), config)
    assert state["positions"]["Event Test"]["shares"] == 60
    assert state["positions"]["Event Test"]["status"] == "exit_pending"
    complete = fill("exit.02", 100, side="SLD", order_ref="SPY|SELL|Event Test|2026-09-01")
    reconcile_event_fills(state, harvest.normalize([entry, complete]), config)
    assert not state["positions"]
    assert state["completed"]


def test_sleeve_symbol_does_not_conflate_contracts():
    from sleeve_fills import signed_inventory
    rows = harvest.normalize([fill(order_ref="SPY|BUY|Trend Sleeve|2026-09-01"),
                              fill("other.01", con_id=43, order_ref="SPY|BUY|Trend Sleeve|2026-09-01")])
    with pytest.raises(RuntimeError, match="multiple contracts"):
        signed_inventory(rows, "Trend Sleeve")


def test_trend_small_band_and_cash_target_keep_actual_inventory(monkeypatch, tmp_path):
    import trend_sleeve as trend
    monkeypatch.setattr(trend, "STATE_LOCAL", str(tmp_path / "state.json"))
    monkeypatch.setattr(trend, "upload_from_local", lambda *a: True)
    monkeypatch.delenv("LOCAL_AUTOMATION_STRICT", raising=False)
    nav = trend.TREND_NAV_FRACTION * trend.ACCOUNT_VALUE
    target = pd.DataFrame([{"Ticker": "SPY", "Eligible": True, "Close": 100., "Weight": 10100/nav, "Asof": "2026-09-04", "Signal": 1}])
    prior = {"positions": {"SPY": {"shares": 100}}, "inventory_basis": "attributed_executions"}
    orders = trend.build_orders(target, prior)
    assert orders.empty
    trend.save_state(target, False, prior_state=prior, orders=orders)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["positions"]["SPY"]["shares"] == saved["expected_positions"]["SPY"]["shares"] == 100
    saved["expected_positions"]["SPY"]["shares"] = 0
    saved["generated"] = "2026-08-01"
    assert report.reconcile_trend(saved, [{"symbol": "SPY", "sec_type": "STK", "position": 100}])


def test_current_degraded_producer_receipt_does_not_erase_success(tmp_path):
    from scripts.automation_supervisor import CommandSpec, Receipt, effective_status
    now = dt.datetime.now(dt.timezone.utc)
    commands = [CommandSpec("scan", ("python", "daily_scan.py", "--scope=all", "--bookend=am"))]
    path = tmp_path / "data/scan_coverage_all_am.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"generated_at": now.isoformat(), "status": "degraded", "unavailable": ["ABC"]}))
    health, detail = inspect_health(commands, tmp_path, now - dt.timedelta(seconds=1))
    assert health == "degraded" and "1" in detail
    receipt = Receipt("v1", "pipeline", "scan", "2026-09-06", "success", "local", "fixture", now.isoformat(), now.isoformat(), health_status=health)
    assert receipt.status == "success"
    assert effective_status(receipt, now) == "degraded"
    health, detail = inspect_health(commands, tmp_path, now + dt.timedelta(seconds=1))
    assert health == "degraded" and "unverified" in detail
