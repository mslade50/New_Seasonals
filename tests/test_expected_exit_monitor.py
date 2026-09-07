import json

from scripts import monitor_expected_exits as monitor


def inputs(now="2026-09-08T20:06:00+00:00", *, remaining=100):
    position = {"tranche_id": "fixture", "account_key": "primary", "account": "TEST_PRIMARY", "con_id": 42,
                "symbol": "SPY", "strategy": "Algo Test", "ref_date": "2026-09-01", "signed_qty": remaining,
                "exit_deadline_utc": "2026-09-08T20:00:00+00:00"}
    inventory = {"status": "known", "asof_utc": now, "tranches": [position] if remaining else []}
    book = {"at": monitor._stamp(now).timestamp() * 1000,
            "accounts": [{"key": "primary", "broker_account": "TEST_PRIMARY", "orders": [],
                          "positions": [{"account": "TEST_PRIMARY", "con_id": 42, "position": remaining}] if remaining else []}]}
    fills = {"fills": [], "completeness": {"accounts": {"primary": {
        "complete": True, "received_at": now, "source_at": now, "broker_account": "TEST_PRIMARY"}}}}
    return inventory, book, fills


def test_five_minute_grace_then_one_deduplicated_missed_event():
    early = "2026-09-08T20:04:00+00:00"
    report, state = monitor.evaluate(*inputs(early), now=early)
    assert report["counts"]["pending"] == 1 and not state["notifications"]
    report, state = monitor.evaluate(*inputs(), state, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["missed"] == 1 and len(state["notifications"]) == 1
    report, state = monitor.evaluate(*inputs(), state, now="2026-09-08T20:06:00+00:00")
    assert len(state["notifications"]) == 1


def test_flat_requires_matching_account_contract_and_closing_fills():
    _, state = monitor.evaluate(*inputs(), now="2026-09-08T20:06:00+00:00")
    inventory, book, fills = inputs(remaining=0)
    wrong = {"account_key": "primary", "account": "TEST_OTHER", "con_id": 42,
             "exec_id": "close.01", "qty": 100, "side": "SLD", "order_ref": "SPY|SELL|Algo Test|2026-09-01"}
    fills["fills"] = [wrong]
    report, _ = monitor.evaluate(inventory, book, fills, state, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1
    fills["fills"] = [dict(wrong, account="TEST_PRIMARY")]
    report, state = monitor.evaluate(inventory, book, fills, state, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["resolved"] == 1
    assert any(row["kind"] == "resolved" for row in state["notifications"].values())


def test_missing_inventory_and_stale_book_do_not_become_fake_zero():
    inventory, book, fills = inputs()
    inventory.update(status="unknown", tranches=[])
    report, _ = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1
    assert report["obligations"][0]["remaining_tagged_qty"] is None
    inventory, book, fills = inputs()
    book["at"] -= 180000
    report, _ = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1


def test_netted_broker_flat_does_not_erase_an_open_algo_tranche():
    inventory, book, fills = inputs()
    book["accounts"][0]["positions"] = []
    report, _ = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    row = report["obligations"][0]
    assert row["broker_net_qty"] == 0 and row["remaining_tagged_qty"] == 100 and row["status"] == "missed"


def test_1610_summary_is_once_per_session():
    now = "2026-09-08T20:10:00+00:00"
    report, state = monitor.evaluate(*inputs(now), now=now)
    assert report["summary_due"]
    report, state = monitor.evaluate(*inputs(now), state, now=now)
    assert sum(row["kind"] == "summary" for row in state["notifications"].values()) == 1


def test_email_ambiguous_failure_is_not_automatically_retried(tmp_path):
    state = {"notifications": {"id": {"status": "pending", "message": "fixture"}}}
    calls = []
    def sender(*args):
        calls.append(args)
        raise TimeoutError("ambiguous SMTP acknowledgement")
    assert not monitor.deliver_pending(state, tmp_path / "state.json", sender)
    assert state["notifications"]["id"]["status"] == "delivery_unknown"
    assert monitor.deliver_pending(state, tmp_path / "state.json", sender)
    assert len(calls) == 1


def test_cli_never_sends_without_explicit_flag(tmp_path, monkeypatch):
    import daily_execution_report
    monkeypatch.setattr(daily_execution_report, "send_email", lambda *a: (_ for _ in ()).throw(AssertionError("unexpected email")))
    paths = {}
    for name, payload in zip(("inventory", "book", "fills"), inputs()):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(payload))
    args = ["--asof", "2026-09-08T20:06:00+00:00", "--state", str(tmp_path / "state.json"), "--output", str(tmp_path / "report.json")]
    for name, path in paths.items():
        args.extend(["--" + name, str(path)])
    assert monitor.main(args) == 0
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "attention"


def test_boolean_false_send_result_is_ambiguous_and_not_retried(tmp_path):
    state = {"notifications": {"id": {"status": "pending", "message": "fixture"}}}
    calls = []
    def sender(*args):
        calls.append(args)
        return False
    assert not monitor.deliver_pending(state, tmp_path / "state.json", sender)
    assert state["notifications"]["id"]["status"] == "delivery_unknown"
    monitor.deliver_pending(state, tmp_path / "state.json", sender)
    assert len(calls) == 1


def test_correction_that_changes_contract_revokes_prior_closing_evidence():
    obligation = dict(inputs()[0]["tranches"][0], baseline_signed_qty=100)
    close = {"account_key": "primary", "account": "TEST_PRIMARY", "con_id": 42, "exec_id": "fixture.01",
             "qty": 100, "side": "SLD", "order_ref": "SPY|SELL|Algo Test|2026-09-01"}
    assert monitor._closing_fills([close, dict(close, exec_id="fixture.02", con_id=43)], obligation) == 0


def test_bad_deadline_emits_unverified_event_and_wrong_fill_account_fails():
    inventory, book, fills = inputs()
    inventory["tranches"][0]["exit_deadline_utc"] = "invalid"
    report, state = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1 and len(state["notifications"]) == 1
    inventory, book, fills = inputs()
    fills["completeness"]["accounts"]["primary"]["broker_account"] = "TEST_OTHER"
    report, _ = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1


def test_reopened_obligation_has_a_new_alert_episode():
    _, state = monitor.evaluate(*inputs(), now="2026-09-08T20:06:00+00:00")
    obligation = next(iter(state["obligations"].values()))
    obligation["last_status"] = "resolved"
    report, state = monitor.evaluate(*inputs(), state, now="2026-09-08T20:06:00+00:00")
    assert report["obligations"][0]["episode"] == 1 and len(state["notifications"]) == 2


def test_naive_timestamp_is_unverified_and_pa_does_not_become_primary():
    inventory, book, fills = inputs()
    fills["completeness"]["accounts"]["primary"]["source_at"] = "2026-09-08T20:06:00"
    report, _ = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1
    inventory, book, fills = inputs()
    inventory["tranches"][0]["account_key"] = "pa"
    report, _ = monitor.evaluate(inventory, book, fills, now="2026-09-08T20:06:00+00:00")
    assert report["counts"]["unable_to_verify"] == 1
